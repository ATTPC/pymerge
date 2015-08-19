import multiprocessing as mp
import logging
import numpy as np
from clint.textui import puts, indent, progress
import os
import glob
import sys
import argparse
import scipy.stats
from functools import reduce
import copy
from collections import deque

sys.path.append('pytpc')
import pytpc


def subtract_and_delete_fpn(traces):
    """Subtract the normalized, average fixed-pattern noise from the data.

    The FPN channels for each AGET are averaged, renormalized to zero, and subtracted
    from the data signals in that AGET. They are then deleted from the data.

    Parameters
    ----------
    traces : np.ndarray
        The structured NumPy array from the event, e.g. `evt.traces`

    Returns
    -------
    ndarray
        The same structured array as before, but with the FPN subtracted and deleted.

    """

    traces = copy.deepcopy(traces)

    fpn_channels = [11, 22, 45, 56]

    for cobo, asad, aget in {tuple(a) for a in traces[['cobo', 'asad', 'aget']]}:
        fpn_idx = np.where(np.all((traces['cobo'] == cobo, traces['asad'] == asad,
                                   traces['aget'] == aget, np.in1d(traces['channel'], fpn_channels)), axis=0))[0]
        data_idx = np.where(np.all((traces['cobo'] == cobo, traces['asad'] == asad,
                                    traces['aget'] == aget, ~np.in1d(traces['channel'], fpn_channels)), axis=0))[0]

        if len(fpn_idx) != 4:
            logger.warn('Number of FPN channels was incorrect: %d (should be 4)' % len(fpn_idx))

        mean_fpn = traces['data'][fpn_idx].mean(axis=0)
        mean_fpn -= mean_fpn.mean()

        traces['data'][data_idx] -= mean_fpn

    return np.delete(traces, np.where(np.in1d(traces['channel'], fpn_channels)))


class EventProcessor(mp.Process):
    def __init__(self, lookup, peds, threshold, inq, outq):
        self.lookup = lookup
        self.peds = peds
        self.threshold = threshold
        self.inq = inq
        self.outq = outq
        super().__init__()

    def process_event(self, evtid, timestamp, rawframes):
        frames = [pytpc.grawdata.GRAWFile._parse(r) for r in rawframes]
        evt = pytpc.Event(evtid, timestamp)
        evt.traces = np.concatenate(frames)

        evt.traces['pad'] = [self.lookup.get(tuple(a), 20000) for a in evt.traces[['cobo', 'asad', 'aget', 'channel']]]

        evt.traces = subtract_and_delete_fpn(evt.traces)

        bad_channels = np.where(evt.traces['pad'] == 20000)[0]
        if len(bad_channels) != 0:
            addr_list = '[' + ', '.join(('{}/{}/{}/{}'.format(evt.traces['cobo'][i], evt.traces['asad'][i],
                                                              evt.traces['aget'][i], evt.traces['channel'][i])
                                                              for i in bad_channels)) + ']'
            logger.warn('Event %d contained unmapped channels: %s', evtid, addr_list)
            evt.traces = np.delete(evt.traces, bad_channels)

        if self.peds is not None:
            evt.traces['data'] = (evt.traces['data'].T - self.peds[evt.traces['pad']]).T

        if self.threshold is not None:
            evt.traces['data'] = scipy.stats.threshold(evt.traces['data'], threshmin=self.threshold)

        return evt

    def run(self):
        while True:
            try:
                logger.debug('Getting input from inq')
                inp = self.inq.get()
                logger.debug('Got input from inq')
                if isinstance(inp, str) and inp == 'STOP':
                    logger.debug('Worker received STOP')
                    break
                else:
                    evtid, timestamp, frames = inp
            except TypeError:
                logger.exception('TypeError in worker when receiving frames')
            except (KeyboardInterrupt, SystemExit):
                logger.debug('Got interrupted. Dying.')
                return
            finally:
                self.inq.task_done()

            try:
                evt = self.process_event(evtid, timestamp, frames)
                logger.debug('Processed event %d', evtid)
            except (KeyboardInterrupt, SystemExit):
                logger.debug('Got interrupted. Dying.')
                return
            except:
                logger.exception('Exception in event processing')
            else:
                logger.debug('Waiting to put event on outq')
                self.outq.put(evt)
                logger.debug('Finished putting event on outq')


class WriterProcess(mp.Process):
    def __init__(self, outq, filename):
        self.filename = filename
        self.outq = outq
        super().__init__()

    def run(self):
        outfile = pytpc.evtdata.EventFile(self.filename, 'w')
        num_events_processed = 0
        while True:
            try:
                logger.debug('Getting output')
                r = self.outq.get()
                logger.debug('Got output')
                if isinstance(r, str) and r == 'STOP':
                    logger.debug('Writer received stop')
                    break
                else:
                    outfile.write(r)
                    logger.debug('Wrote event %d', r.evt_id)
            except (KeyboardInterrupt, SystemExit):
                logger.debug('Got interrupted. Dying.')
                return
            finally:
                self.outq.task_done()
                num_events_processed += 1
        outfile.close()

def main():

    # Parse the command line options

    parser = argparse.ArgumentParser(description='GRAW Frame merger')
    parser.add_argument('input', type=str,
                        help='Path to a directory of GRAW files')
    parser.add_argument('output', type=str, nargs='?',
                        help='Path to the output file')
    parser.add_argument('--lookup', '-l', type=str,
                        help='Path to the pad lookup table, as a CSV file')
    parser.add_argument('--pedestals', '-p', type=str,
                        help='Path to a table of pedestal values')
    parser.add_argument('--threshold', '-t', type=int,
                        help='Threshold value')
    parser.add_argument('--zerosupp', '-z', action='store_true',
                        help='Enable zero suppression in the output file')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Print more information')
    args = parser.parse_args()

    if args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARN)

    # Find and map the GRAW files

    print('Looking for files')
    gfile_paths = glob.glob(os.path.join(os.path.abspath(args.input), '*.graw'))
    with indent(4):
        for x in gfile_paths:
            puts('Found file: %s' % os.path.basename(x))
    print('Found {} GRAW files'.format(len(gfile_paths)))

    print('Mapping frames in files')

    for path in progress.bar(gfile_paths):
        gf = pytpc.grawdata.GRAWFile(path)
        gf.close()

    # Load the lookup table and pedestals, and parse the threshold option

    if args.lookup is not None:
        print('Loading lookup table')
        lookup = {}
        with open(args.lookup) as lfile:
            for line in lfile:
                cobo, asad, aget, ch, pad = [int(a) for a in line.strip().split(',')]
                if -1 in (cobo, asad, aget, ch, pad):
                    continue
                lookup[(cobo, asad, aget, ch)] = pad
    else:
        lookup = {}

    if args.pedestals is not None:
        print('Loading pedestals')
        peds = np.zeros(10240)
        with open(args.pedestals) as pfile:
            for line in pfile:
                pad, ped = line.strip().split(',')
                peds[int(pad)] = float(ped) if '.' in ped else int(ped)
    else:
        peds = None

    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = None

    # Now open the files for reading

    gfiles = [pytpc.grawdata.GRAWFile(g) for g in gfile_paths]
    valid_events = sorted(reduce(set.union, [set(g.evtids) for g in gfiles]))  # TODO: Make this check for frames that are in all cobos

    logger.debug('Length of set of valid events was %d', len(valid_events))

    if args.output is not None:
        output_name = args.output
    else:
        run_name = os.path.split(os.path.normpath(args.input))[-1]
        output_name = run_name + '.evt'

    # Prepare the queues and worker/writer processes

    logger.debug('Making queues')

    inq = mp.JoinableQueue(50)
    outq = mp.JoinableQueue(50)

    logger.debug('Making processes')

    workers = [EventProcessor(lookup, peds, threshold, inq, outq) for i in range(8)]
    for w in workers:
        w.start()
    writer = WriterProcess(outq, output_name)
    writer.start()

    # Now start putting frames on the input queue

    print('Beginning merge')

    for i in progress.bar(valid_events):
        frames = []
        for g in gfiles:
            this_frames = g.get_raw_frames_for_event(i)
            frames += this_frames

        if len(frames) == 0:
            logger.warn('No frames in event %d', i)
            continue
        else:
            logging.debug('Found %d frames for event %d', len(frames), i)

        inq.put((i, 0, frames))
        logger.debug('Put in frames for event %d', i)

    # Wrap up the queues and worker/writer processes

    logger.debug('Waiting for all jobs to finish')
    inq.join()  # blocks main thread until queue is empty and all tasks on it are finished
    outq.join()

    for i in range(len(workers)):
        inq.put('STOP')  # This tells the processes to die
    outq.put('STOP')

    for proc in workers:
        proc.join()
    writer.join()


if __name__ == '__main__':
    logger = mp.log_to_stderr()
    main()
