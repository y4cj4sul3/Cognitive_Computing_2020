import os
import pickle
import shutil

import time
from datetime import datetime

from UltimakerPrinter import Printer

printerName = 'S5'
printer = Printer(printerName)

while True:
    # check printer state every minute
    print('wait for printing...')
    while printer.getPrintJobState() != 'printing':
        time.sleep(60)

    print("start printing!")

    # get print job info
    printJob = printer.getPrintJob()
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    folderPath = 'data/UM{}/{}_{}/'.format(printerName, printJob['name'], date)
    os.makedirs(folderPath)
    print(folderPath)
    with open(folderPath + 'printJob_start.pkl', 'wb') as fp:
        pickle.dump(printJob, fp, )

    # get gcode
    gcode = printer.getPrintJobGcode()
    with open(folderPath + 'path.gcode', 'w') as fp:
        fp.write(gcode)

    progress = []
    timestamp = []
    # snapshot during printing
    while True:
        # current progress and time
        progress.append(printer.getPrintJobProgress())
        timestamp.append(datetime.now().timestamp())
        print(timestamp[-1], progress[-1])
        # get snaphot
        img = printer.getCameraSnapshot()
        if img is not None and progress[-1] is not None:
            with open(folderPath + str(timestamp[-1]) + '.png', 'wb') as fp:
                shutil.copyfileobj(img, fp)
                
        if progress[-1] == 1:
            break

    # progress data
    progress_data = {
        'timestamp': timestamp,
        'progress': progress
    }
    with open(folderPath + 'progress.pkl', 'wb') as fp:
        pickle.dump(progress_data, fp)

    # print job
    printJob = printer.getPrintJob()
    with open(folderPath + 'printJob_finish.pkl', 'wb') as fp:
        pickle.dump(printJob, fp)

    print('print finished !')

