# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys

from PySide2.QtGui import QGuiApplication
from PySide2.QtQml import QQmlApplicationEngine
from PySide2.QtCore import QObject, Slot, Signal
from object_tracker import runner
from object_tracker_webcam import runner_webcam

class MainWindow(QObject):
    def __init__(self):
        QObject.__init__(self)

    #Set value
    saveValueSignal = Signal(int)
    getValueSignal = Signal(int)
    runCCTVSignal = Signal(int)
    runWebcamSignal = Signal(int)

    #Function set

    @Slot(int)
    def runCCTV(self, value):
        try:
            runner()
        except:
            pass
        self.runCCTVSignal.emit(1)

    @Slot(int)
    def runWebcam(self, value):
        try:
            runner_webcam()
        except:
            pass
        self.runWebcamSignal.emit(1)

    @Slot(int)
    def getValue(self, value):
        f = open("config.txt", "r+")
        threshold = int(f.readline())
        f.close()
        self.getValueSignal.emit(threshold)

    @Slot(int)
    def saveValue(self, thresh_value):
        f = open("config.txt", "r+")
        f.truncate(0)
        f.write(str(thresh_value))
        f.close()
        self.saveValueSignal.emit(1)

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    #Get Context
    main = MainWindow()
    engine.rootContext().setContextProperty("backend", main)

    #Load QML file
    engine.load(os.fspath(Path(__file__).resolve().parent / "qml/main.qml"))

    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec_())
