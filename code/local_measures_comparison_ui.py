# Form implementation generated from reading ui file 'C:\Users\yonat\PyCharmMiscProject\local_measures_comparison.ui'
#
# Created by: PyQt6 UI code generator 6.8.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_LocalMeasureComparisonWindow(object):
    def setupUi(self, LocalMeasureComparisonWindow):
        LocalMeasureComparisonWindow.setObjectName("LocalMeasureComparisonWindow")
        LocalMeasureComparisonWindow.resize(1200, 880)
        LocalMeasureComparisonWindow.setMaximumSize(QtCore.QSize(1200, 880))
        LocalMeasureComparisonWindow.setStyleSheet("background-color: #e0ffff ;")
        self.centralwidget = QtWidgets.QWidget(parent=LocalMeasureComparisonWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.headerLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.headerLabel.setGeometry(QtCore.QRect(380, 40, 431, 40))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.headerLabel.setFont(font)
        self.headerLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.headerLabel.setObjectName("headerLabel")
        self.groupLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.groupLabel.setGeometry(QtCore.QRect(180, 120, 100, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.groupLabel.setFont(font)
        self.groupLabel.setObjectName("groupLabel")
        self.groupListWidget = QtWidgets.QListWidget(parent=self.centralwidget)
        self.groupListWidget.setGeometry(QtCore.QRect(140, 150, 150, 200))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.groupListWidget.setFont(font)
        self.groupListWidget.setStyleSheet("background-color: white ;")
        self.groupListWidget.setObjectName("groupListWidget")
        self.conditionLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.conditionLabel.setGeometry(QtCore.QRect(355, 120, 100, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.conditionLabel.setFont(font)
        self.conditionLabel.setObjectName("conditionLabel")
        self.conditionListWidget = QtWidgets.QListWidget(parent=self.centralwidget)
        self.conditionListWidget.setGeometry(QtCore.QRect(330, 150, 150, 200))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.conditionListWidget.setFont(font)
        self.conditionListWidget.setStyleSheet("background-color: white ;")
        self.conditionListWidget.setObjectName("conditionListWidget")
        self.dyadLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.dyadLabel.setGeometry(QtCore.QRect(565, 120, 100, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.dyadLabel.setFont(font)
        self.dyadLabel.setObjectName("dyadLabel")
        self.dyadListWidget = QtWidgets.QListWidget(parent=self.centralwidget)
        self.dyadListWidget.setGeometry(QtCore.QRect(520, 150, 150, 200))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.dyadListWidget.setFont(font)
        self.dyadListWidget.setStyleSheet("background-color: white ;")
        self.dyadListWidget.setObjectName("dyadListWidget")
        self.selectAllDyadsButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.selectAllDyadsButton.setGeometry(QtCore.QRect(510, 360, 100, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.selectAllDyadsButton.setFont(font)
        self.selectAllDyadsButton.setStyleSheet("background-color: white ;")
        self.selectAllDyadsButton.setObjectName("selectAllDyadsButton")
        self.clearDyadsButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.clearDyadsButton.setGeometry(QtCore.QRect(610, 360, 70, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.clearDyadsButton.setFont(font)
        self.clearDyadsButton.setStyleSheet("background-color: white ;")
        self.clearDyadsButton.setObjectName("clearDyadsButton")
        self.metricLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.metricLabel.setGeometry(QtCore.QRect(975, 120, 111, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.metricLabel.setFont(font)
        self.metricLabel.setObjectName("metricLabel")
        self.metricComboBox = QtWidgets.QComboBox(parent=self.centralwidget)
        self.metricComboBox.setGeometry(QtCore.QRect(930, 150, 200, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.metricComboBox.setFont(font)
        self.metricComboBox.setStyleSheet("background-color: white ;")
        self.metricComboBox.setObjectName("metricComboBox")
        self.metricComboBox.addItem("")
        self.metricComboBox.addItem("")
        self.localMeasureLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.localMeasureLabel.setGeometry(QtCore.QRect(770, 120, 150, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.localMeasureLabel.setFont(font)
        self.localMeasureLabel.setObjectName("localMeasureLabel")
        self.nodeMeasureListWidget = QtWidgets.QListWidget(parent=self.centralwidget)
        self.nodeMeasureListWidget.setGeometry(QtCore.QRect(710, 150, 180, 200))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.nodeMeasureListWidget.setFont(font)
        self.nodeMeasureListWidget.setStyleSheet("background-color: white ;")
        self.nodeMeasureListWidget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.nodeMeasureListWidget.setObjectName("nodeMeasureListWidget")
        self.selectAllNodesButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.selectAllNodesButton.setGeometry(QtCore.QRect(716, 360, 100, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.selectAllNodesButton.setFont(font)
        self.selectAllNodesButton.setStyleSheet("background-color: white ;")
        self.selectAllNodesButton.setObjectName("selectAllNodesButton")
        self.clearNodesButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.clearNodesButton.setGeometry(QtCore.QRect(816, 360, 70, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.clearNodesButton.setFont(font)
        self.clearNodesButton.setStyleSheet("background-color: white ;")
        self.clearNodesButton.setObjectName("clearNodesButton")
        self.chartTypeLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.chartTypeLabel.setGeometry(QtCore.QRect(975, 200, 111, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.chartTypeLabel.setFont(font)
        self.chartTypeLabel.setObjectName("chartTypeLabel")
        self.chartTypeComboBox = QtWidgets.QComboBox(parent=self.centralwidget)
        self.chartTypeComboBox.setGeometry(QtCore.QRect(930, 230, 200, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.chartTypeComboBox.setFont(font)
        self.chartTypeComboBox.setStyleSheet("background-color: white ;")
        self.chartTypeComboBox.setObjectName("chartTypeComboBox")
        self.chartTypeComboBox.addItem("")
        self.chartTypeComboBox.addItem("")
        self.openPlotImageButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.openPlotImageButton.setGeometry(QtCore.QRect(930, 381, 120, 35))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.openPlotImageButton.setFont(font)
        self.openPlotImageButton.setStyleSheet("background-color: white ;")
        self.openPlotImageButton.setObjectName("openPlotImageButton")
        self.compareButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.compareButton.setGeometry(QtCore.QRect(960, 280, 140, 40))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.compareButton.setFont(font)
        self.compareButton.setStyleSheet("background-color: white ;")
        self.compareButton.setObjectName("compareButton")
        self.plotWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.plotWidget.setGeometry(QtCore.QRect(150, 415, 905, 400))
        self.plotWidget.setObjectName("plotWidget")
        self.plotLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.plotLabel.setGeometry(QtCore.QRect(150, 419, 900, 381))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.plotLabel.setFont(font)
        self.plotLabel.setStyleSheet("background-color: white ;")
        self.plotLabel.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.plotLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.plotLabel.setObjectName("plotLabel")
        self.openCsvButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.openCsvButton.setGeometry(QtCore.QRect(1020, 50, 120, 35))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.openCsvButton.setFont(font)
        self.openCsvButton.setStyleSheet("background-color: white ;")
        self.openCsvButton.setObjectName("openCsvButton")
        self.helpButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.helpButton.setGeometry(QtCore.QRect(20, 90, 100, 35))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.helpButton.setFont(font)
        self.helpButton.setStyleSheet("background-color: white ;")
        self.helpButton.setObjectName("helpButton")
        self.backButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.backButton.setGeometry(QtCore.QRect(20, 40, 100, 35))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.backButton.setFont(font)
        self.backButton.setStyleSheet("background-color: white ;")
        self.backButton.setObjectName("backButton")
        self.exportButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.exportButton.setGeometry(QtCore.QRect(970, 330, 120, 35))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.exportButton.setFont(font)
        self.exportButton.setStyleSheet("background-color: white ;")
        self.exportButton.setObjectName("exportButton")
        LocalMeasureComparisonWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=LocalMeasureComparisonWindow)
        self.statusbar.setObjectName("statusbar")
        LocalMeasureComparisonWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(parent=LocalMeasureComparisonWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 27))
        self.menubar.setObjectName("menubar")
        LocalMeasureComparisonWindow.setMenuBar(self.menubar)

        self.retranslateUi(LocalMeasureComparisonWindow)
        QtCore.QMetaObject.connectSlotsByName(LocalMeasureComparisonWindow)

    def retranslateUi(self, LocalMeasureComparisonWindow):
        _translate = QtCore.QCoreApplication.translate
        LocalMeasureComparisonWindow.setWindowTitle(_translate("LocalMeasureComparisonWindow", "Local Measures Comparison"))
        self.headerLabel.setText(_translate("LocalMeasureComparisonWindow", "Local Graph Measures Comparison"))
        self.groupLabel.setText(_translate("LocalMeasureComparisonWindow", "Groups"))
        self.conditionLabel.setText(_translate("LocalMeasureComparisonWindow", "Conditions"))
        self.dyadLabel.setText(_translate("LocalMeasureComparisonWindow", "Dyads"))
        self.selectAllDyadsButton.setText(_translate("LocalMeasureComparisonWindow", "Select All"))
        self.clearDyadsButton.setText(_translate("LocalMeasureComparisonWindow", "Clear"))
        self.metricLabel.setText(_translate("LocalMeasureComparisonWindow", "Local Metric"))
        self.metricComboBox.setItemText(0, _translate("LocalMeasureComparisonWindow", "Node Strength"))
        self.metricComboBox.setItemText(1, _translate("LocalMeasureComparisonWindow", "Local Efficiency"))
        self.localMeasureLabel.setText(_translate("LocalMeasureComparisonWindow", "Nodes"))
        self.selectAllNodesButton.setText(_translate("LocalMeasureComparisonWindow", "Select All"))
        self.clearNodesButton.setText(_translate("LocalMeasureComparisonWindow", "Clear"))
        self.chartTypeLabel.setText(_translate("LocalMeasureComparisonWindow", "Chart Type"))
        self.chartTypeComboBox.setItemText(0, _translate("LocalMeasureComparisonWindow", "Bar Chart"))
        self.chartTypeComboBox.setItemText(1, _translate("LocalMeasureComparisonWindow", "Line Chart"))
        self.openPlotImageButton.setText(_translate("LocalMeasureComparisonWindow", "🖼️ View"))
        self.compareButton.setText(_translate("LocalMeasureComparisonWindow", "📊 Compare"))
        self.plotLabel.setText(_translate("LocalMeasureComparisonWindow", "Comparison chart will appear here"))
        self.openCsvButton.setText(_translate("LocalMeasureComparisonWindow", "Open CSV"))
        self.helpButton.setText(_translate("LocalMeasureComparisonWindow", "💡 Help"))
        self.backButton.setText(_translate("LocalMeasureComparisonWindow", "⬅️ Back"))
        self.exportButton.setText(_translate("LocalMeasureComparisonWindow", "💾 Export"))
