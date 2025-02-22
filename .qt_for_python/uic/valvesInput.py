# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Kula\Petrobras\OpenPulse\data\user_input\ui\Model\Setup\Structural\valvesInput.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(500, 480)
        Dialog.setMinimumSize(QtCore.QSize(500, 458))
        Dialog.setMaximumSize(QtCore.QSize(500, 480))
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setGeometry(QtCore.QRect(0, 58, 500, 422))
        self.frame.setMinimumSize(QtCore.QSize(500, 422))
        self.frame.setMaximumSize(QtCore.QSize(500, 422))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setObjectName("frame")
        self.pushButton_confirm = QtWidgets.QPushButton(self.frame)
        self.pushButton_confirm.setGeometry(QtCore.QRect(166, 374, 169, 36))
        self.pushButton_confirm.setMinimumSize(QtCore.QSize(169, 36))
        self.pushButton_confirm.setMaximumSize(QtCore.QSize(169, 36))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.pushButton_confirm.setFont(font)
        self.pushButton_confirm.setAutoDefault(True)
        self.pushButton_confirm.setDefault(False)
        self.pushButton_confirm.setFlat(False)
        self.pushButton_confirm.setObjectName("pushButton_confirm")
        self.label_selected_id = QtWidgets.QLabel(self.frame)
        self.label_selected_id.setEnabled(True)
        self.label_selected_id.setGeometry(QtCore.QRect(36, 18, 152, 30))
        self.label_selected_id.setMinimumSize(QtCore.QSize(152, 30))
        self.label_selected_id.setMaximumSize(QtCore.QSize(152, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_selected_id.setFont(font)
        self.label_selected_id.setMouseTracking(True)
        self.label_selected_id.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_selected_id.setObjectName("label_selected_id")
        self.lineEdit_selected_ID = QtWidgets.QLineEdit(self.frame)
        self.lineEdit_selected_ID.setEnabled(True)
        self.lineEdit_selected_ID.setGeometry(QtCore.QRect(192, 18, 200, 30))
        self.lineEdit_selected_ID.setMinimumSize(QtCore.QSize(200, 30))
        self.lineEdit_selected_ID.setMaximumSize(QtCore.QSize(200, 30))
        self.lineEdit_selected_ID.setSizeIncrement(QtCore.QSize(0, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.lineEdit_selected_ID.setFont(font)
        self.lineEdit_selected_ID.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_selected_ID.setStyleSheet("color: rgb(255, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.lineEdit_selected_ID.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_selected_ID.setObjectName("lineEdit_selected_ID")
        self.tabWidget_inputs = QtWidgets.QTabWidget(self.frame)
        self.tabWidget_inputs.setEnabled(True)
        self.tabWidget_inputs.setGeometry(QtCore.QRect(26, 134, 449, 230))
        self.tabWidget_inputs.setMinimumSize(QtCore.QSize(449, 230))
        self.tabWidget_inputs.setMaximumSize(QtCore.QSize(449, 230))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.tabWidget_inputs.setFont(font)
        self.tabWidget_inputs.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget_inputs.setDocumentMode(False)
        self.tabWidget_inputs.setTabsClosable(False)
        self.tabWidget_inputs.setMovable(False)
        self.tabWidget_inputs.setTabBarAutoHide(False)
        self.tabWidget_inputs.setObjectName("tabWidget_inputs")
        self.tab_insert_by_line = QtWidgets.QWidget()
        self.tab_insert_by_line.setEnabled(True)
        self.tab_insert_by_line.setObjectName("tab_insert_by_line")
        self.lineEdit_valve_length_line = QtWidgets.QLineEdit(self.tab_insert_by_line)
        self.lineEdit_valve_length_line.setEnabled(False)
        self.lineEdit_valve_length_line.setGeometry(QtCore.QRect(190, 28, 120, 30))
        self.lineEdit_valve_length_line.setMinimumSize(QtCore.QSize(120, 30))
        self.lineEdit_valve_length_line.setMaximumSize(QtCore.QSize(120, 30))
        self.lineEdit_valve_length_line.setSizeIncrement(QtCore.QSize(0, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.lineEdit_valve_length_line.setFont(font)
        self.lineEdit_valve_length_line.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_valve_length_line.setStyleSheet("color: rgb(0, 0, 255);\n"
"background-color: rgb(255, 255, 255);")
        self.lineEdit_valve_length_line.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_valve_length_line.setObjectName("lineEdit_valve_length_line")
        self.label_selected_id_9 = QtWidgets.QLabel(self.tab_insert_by_line)
        self.label_selected_id_9.setEnabled(True)
        self.label_selected_id_9.setGeometry(QtCore.QRect(22, 28, 160, 30))
        self.label_selected_id_9.setMinimumSize(QtCore.QSize(160, 30))
        self.label_selected_id_9.setMaximumSize(QtCore.QSize(160, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_selected_id_9.setFont(font)
        self.label_selected_id_9.setMouseTracking(True)
        self.label_selected_id_9.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_selected_id_9.setObjectName("label_selected_id_9")
        self.label_109 = QtWidgets.QLabel(self.tab_insert_by_line)
        self.label_109.setEnabled(True)
        self.label_109.setGeometry(QtCore.QRect(314, 28, 40, 30))
        self.label_109.setMinimumSize(QtCore.QSize(40, 30))
        self.label_109.setMaximumSize(QtCore.QSize(40, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_109.setFont(font)
        self.label_109.setMouseTracking(True)
        self.label_109.setAlignment(QtCore.Qt.AlignCenter)
        self.label_109.setObjectName("label_109")
        self.lineEdit_valve_mass_line = QtWidgets.QLineEdit(self.tab_insert_by_line)
        self.lineEdit_valve_mass_line.setEnabled(True)
        self.lineEdit_valve_mass_line.setGeometry(QtCore.QRect(190, 140, 120, 30))
        self.lineEdit_valve_mass_line.setMinimumSize(QtCore.QSize(120, 30))
        self.lineEdit_valve_mass_line.setMaximumSize(QtCore.QSize(120, 30))
        self.lineEdit_valve_mass_line.setSizeIncrement(QtCore.QSize(0, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.lineEdit_valve_mass_line.setFont(font)
        self.lineEdit_valve_mass_line.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_valve_mass_line.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.lineEdit_valve_mass_line.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_valve_mass_line.setObjectName("lineEdit_valve_mass_line")
        self.label_selected_id_13 = QtWidgets.QLabel(self.tab_insert_by_line)
        self.label_selected_id_13.setEnabled(True)
        self.label_selected_id_13.setGeometry(QtCore.QRect(22, 140, 160, 30))
        self.label_selected_id_13.setMinimumSize(QtCore.QSize(160, 30))
        self.label_selected_id_13.setMaximumSize(QtCore.QSize(160, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_selected_id_13.setFont(font)
        self.label_selected_id_13.setMouseTracking(True)
        self.label_selected_id_13.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_selected_id_13.setObjectName("label_selected_id_13")
        self.label_108 = QtWidgets.QLabel(self.tab_insert_by_line)
        self.label_108.setEnabled(True)
        self.label_108.setGeometry(QtCore.QRect(314, 140, 40, 30))
        self.label_108.setMinimumSize(QtCore.QSize(40, 30))
        self.label_108.setMaximumSize(QtCore.QSize(40, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_108.setFont(font)
        self.label_108.setMouseTracking(True)
        self.label_108.setAlignment(QtCore.Qt.AlignCenter)
        self.label_108.setObjectName("label_108")
        self.label_112 = QtWidgets.QLabel(self.tab_insert_by_line)
        self.label_112.setEnabled(True)
        self.label_112.setGeometry(QtCore.QRect(314, 84, 40, 30))
        self.label_112.setMinimumSize(QtCore.QSize(40, 30))
        self.label_112.setMaximumSize(QtCore.QSize(40, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_112.setFont(font)
        self.label_112.setMouseTracking(True)
        self.label_112.setAlignment(QtCore.Qt.AlignCenter)
        self.label_112.setObjectName("label_112")
        self.label_selected_id_14 = QtWidgets.QLabel(self.tab_insert_by_line)
        self.label_selected_id_14.setEnabled(True)
        self.label_selected_id_14.setGeometry(QtCore.QRect(22, 84, 160, 30))
        self.label_selected_id_14.setMinimumSize(QtCore.QSize(160, 30))
        self.label_selected_id_14.setMaximumSize(QtCore.QSize(160, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_selected_id_14.setFont(font)
        self.label_selected_id_14.setMouseTracking(True)
        self.label_selected_id_14.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_selected_id_14.setObjectName("label_selected_id_14")
        self.lineEdit_stiffening_factor_line = QtWidgets.QLineEdit(self.tab_insert_by_line)
        self.lineEdit_stiffening_factor_line.setEnabled(True)
        self.lineEdit_stiffening_factor_line.setGeometry(QtCore.QRect(190, 84, 120, 30))
        self.lineEdit_stiffening_factor_line.setMinimumSize(QtCore.QSize(120, 30))
        self.lineEdit_stiffening_factor_line.setMaximumSize(QtCore.QSize(120, 30))
        self.lineEdit_stiffening_factor_line.setSizeIncrement(QtCore.QSize(0, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.lineEdit_stiffening_factor_line.setFont(font)
        self.lineEdit_stiffening_factor_line.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_stiffening_factor_line.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.lineEdit_stiffening_factor_line.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_stiffening_factor_line.setObjectName("lineEdit_stiffening_factor_line")
        self.tabWidget_inputs.addTab(self.tab_insert_by_line, "")
        self.tab_insert_by_element = QtWidgets.QWidget()
        self.tab_insert_by_element.setObjectName("tab_insert_by_element")
        self.lineEdit_valve_mass_element = QtWidgets.QLineEdit(self.tab_insert_by_element)
        self.lineEdit_valve_mass_element.setEnabled(True)
        self.lineEdit_valve_mass_element.setGeometry(QtCore.QRect(190, 140, 120, 30))
        self.lineEdit_valve_mass_element.setMinimumSize(QtCore.QSize(120, 30))
        self.lineEdit_valve_mass_element.setMaximumSize(QtCore.QSize(120, 30))
        self.lineEdit_valve_mass_element.setSizeIncrement(QtCore.QSize(0, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.lineEdit_valve_mass_element.setFont(font)
        self.lineEdit_valve_mass_element.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_valve_mass_element.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.lineEdit_valve_mass_element.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_valve_mass_element.setObjectName("lineEdit_valve_mass_element")
        self.label_107 = QtWidgets.QLabel(self.tab_insert_by_element)
        self.label_107.setEnabled(True)
        self.label_107.setGeometry(QtCore.QRect(314, 140, 40, 30))
        self.label_107.setMinimumSize(QtCore.QSize(40, 30))
        self.label_107.setMaximumSize(QtCore.QSize(40, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_107.setFont(font)
        self.label_107.setMouseTracking(True)
        self.label_107.setAlignment(QtCore.Qt.AlignCenter)
        self.label_107.setObjectName("label_107")
        self.label_104 = QtWidgets.QLabel(self.tab_insert_by_element)
        self.label_104.setEnabled(True)
        self.label_104.setGeometry(QtCore.QRect(314, 28, 40, 30))
        self.label_104.setMinimumSize(QtCore.QSize(40, 30))
        self.label_104.setMaximumSize(QtCore.QSize(40, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_104.setFont(font)
        self.label_104.setMouseTracking(True)
        self.label_104.setAlignment(QtCore.Qt.AlignCenter)
        self.label_104.setObjectName("label_104")
        self.label_selected_id_7 = QtWidgets.QLabel(self.tab_insert_by_element)
        self.label_selected_id_7.setEnabled(True)
        self.label_selected_id_7.setGeometry(QtCore.QRect(22, 140, 160, 30))
        self.label_selected_id_7.setMinimumSize(QtCore.QSize(160, 30))
        self.label_selected_id_7.setMaximumSize(QtCore.QSize(160, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_selected_id_7.setFont(font)
        self.label_selected_id_7.setMouseTracking(True)
        self.label_selected_id_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_selected_id_7.setObjectName("label_selected_id_7")
        self.label_selected_id_12 = QtWidgets.QLabel(self.tab_insert_by_element)
        self.label_selected_id_12.setEnabled(True)
        self.label_selected_id_12.setGeometry(QtCore.QRect(22, 28, 160, 30))
        self.label_selected_id_12.setMinimumSize(QtCore.QSize(160, 30))
        self.label_selected_id_12.setMaximumSize(QtCore.QSize(160, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_selected_id_12.setFont(font)
        self.label_selected_id_12.setMouseTracking(True)
        self.label_selected_id_12.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_selected_id_12.setObjectName("label_selected_id_12")
        self.lineEdit_valve_length_element = QtWidgets.QLineEdit(self.tab_insert_by_element)
        self.lineEdit_valve_length_element.setEnabled(True)
        self.lineEdit_valve_length_element.setGeometry(QtCore.QRect(190, 28, 120, 30))
        self.lineEdit_valve_length_element.setMinimumSize(QtCore.QSize(120, 30))
        self.lineEdit_valve_length_element.setMaximumSize(QtCore.QSize(120, 30))
        self.lineEdit_valve_length_element.setSizeIncrement(QtCore.QSize(0, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.lineEdit_valve_length_element.setFont(font)
        self.lineEdit_valve_length_element.setMouseTracking(True)
        self.lineEdit_valve_length_element.setTabletTracking(False)
        self.lineEdit_valve_length_element.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_valve_length_element.setToolTipDuration(-1)
        self.lineEdit_valve_length_element.setStatusTip("")
        self.lineEdit_valve_length_element.setWhatsThis("")
        self.lineEdit_valve_length_element.setAccessibleDescription("")
        self.lineEdit_valve_length_element.setStyleSheet("color: rgb(0, 0, 255);\n"
"background-color: rgb(255, 255, 255);")
        self.lineEdit_valve_length_element.setInputMask("")
        self.lineEdit_valve_length_element.setText("")
        self.lineEdit_valve_length_element.setFrame(True)
        self.lineEdit_valve_length_element.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.lineEdit_valve_length_element.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_valve_length_element.setDragEnabled(False)
        self.lineEdit_valve_length_element.setObjectName("lineEdit_valve_length_element")
        self.label_111 = QtWidgets.QLabel(self.tab_insert_by_element)
        self.label_111.setEnabled(True)
        self.label_111.setGeometry(QtCore.QRect(314, 84, 40, 30))
        self.label_111.setMinimumSize(QtCore.QSize(40, 30))
        self.label_111.setMaximumSize(QtCore.QSize(40, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_111.setFont(font)
        self.label_111.setMouseTracking(True)
        self.label_111.setAlignment(QtCore.Qt.AlignCenter)
        self.label_111.setObjectName("label_111")
        self.lineEdit_stiffening_factor_element = QtWidgets.QLineEdit(self.tab_insert_by_element)
        self.lineEdit_stiffening_factor_element.setEnabled(True)
        self.lineEdit_stiffening_factor_element.setGeometry(QtCore.QRect(190, 84, 120, 30))
        self.lineEdit_stiffening_factor_element.setMinimumSize(QtCore.QSize(120, 30))
        self.lineEdit_stiffening_factor_element.setMaximumSize(QtCore.QSize(120, 30))
        self.lineEdit_stiffening_factor_element.setSizeIncrement(QtCore.QSize(0, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.lineEdit_stiffening_factor_element.setFont(font)
        self.lineEdit_stiffening_factor_element.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_stiffening_factor_element.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.lineEdit_stiffening_factor_element.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_stiffening_factor_element.setObjectName("lineEdit_stiffening_factor_element")
        self.label_selected_id_8 = QtWidgets.QLabel(self.tab_insert_by_element)
        self.label_selected_id_8.setEnabled(True)
        self.label_selected_id_8.setGeometry(QtCore.QRect(22, 84, 160, 30))
        self.label_selected_id_8.setMinimumSize(QtCore.QSize(160, 30))
        self.label_selected_id_8.setMaximumSize(QtCore.QSize(160, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_selected_id_8.setFont(font)
        self.label_selected_id_8.setMouseTracking(True)
        self.label_selected_id_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_selected_id_8.setObjectName("label_selected_id_8")
        self.tabWidget_inputs.addTab(self.tab_insert_by_element, "")
        self.tab_flange_setup = QtWidgets.QWidget()
        self.tab_flange_setup.setObjectName("tab_flange_setup")
        self.label_128 = QtWidgets.QLabel(self.tab_flange_setup)
        self.label_128.setEnabled(True)
        self.label_128.setGeometry(QtCore.QRect(314, 140, 40, 30))
        self.label_128.setMinimumSize(QtCore.QSize(40, 30))
        self.label_128.setMaximumSize(QtCore.QSize(40, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_128.setFont(font)
        self.label_128.setMouseTracking(True)
        self.label_128.setAlignment(QtCore.Qt.AlignCenter)
        self.label_128.setObjectName("label_128")
        self.lineEdit_outer_diameter = QtWidgets.QLineEdit(self.tab_flange_setup)
        self.lineEdit_outer_diameter.setEnabled(True)
        self.lineEdit_outer_diameter.setGeometry(QtCore.QRect(190, 140, 120, 30))
        self.lineEdit_outer_diameter.setMinimumSize(QtCore.QSize(120, 30))
        self.lineEdit_outer_diameter.setMaximumSize(QtCore.QSize(120, 30))
        self.lineEdit_outer_diameter.setSizeIncrement(QtCore.QSize(0, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.lineEdit_outer_diameter.setFont(font)
        self.lineEdit_outer_diameter.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_outer_diameter.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.lineEdit_outer_diameter.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_outer_diameter.setObjectName("lineEdit_outer_diameter")
        self.label_outer_diameter = QtWidgets.QLabel(self.tab_flange_setup)
        self.label_outer_diameter.setEnabled(True)
        self.label_outer_diameter.setGeometry(QtCore.QRect(20, 140, 160, 30))
        self.label_outer_diameter.setMinimumSize(QtCore.QSize(160, 30))
        self.label_outer_diameter.setMaximumSize(QtCore.QSize(160, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_outer_diameter.setFont(font)
        self.label_outer_diameter.setMouseTracking(True)
        self.label_outer_diameter.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_outer_diameter.setObjectName("label_outer_diameter")
        self.label_selected_id_23 = QtWidgets.QLabel(self.tab_flange_setup)
        self.label_selected_id_23.setEnabled(True)
        self.label_selected_id_23.setGeometry(QtCore.QRect(20, 84, 160, 30))
        self.label_selected_id_23.setMinimumSize(QtCore.QSize(160, 30))
        self.label_selected_id_23.setMaximumSize(QtCore.QSize(160, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_selected_id_23.setFont(font)
        self.label_selected_id_23.setMouseTracking(True)
        self.label_selected_id_23.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_selected_id_23.setObjectName("label_selected_id_23")
        self.label_selected_id_24 = QtWidgets.QLabel(self.tab_flange_setup)
        self.label_selected_id_24.setEnabled(True)
        self.label_selected_id_24.setGeometry(QtCore.QRect(40, 28, 170, 30))
        self.label_selected_id_24.setMinimumSize(QtCore.QSize(170, 30))
        self.label_selected_id_24.setMaximumSize(QtCore.QSize(170, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_selected_id_24.setFont(font)
        self.label_selected_id_24.setMouseTracking(True)
        self.label_selected_id_24.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_selected_id_24.setObjectName("label_selected_id_24")
        self.spinBox_number_elements_flange = QtWidgets.QSpinBox(self.tab_flange_setup)
        self.spinBox_number_elements_flange.setGeometry(QtCore.QRect(218, 28, 65, 30))
        self.spinBox_number_elements_flange.setMinimumSize(QtCore.QSize(0, 30))
        self.spinBox_number_elements_flange.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.spinBox_number_elements_flange.setFont(font)
        self.spinBox_number_elements_flange.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_number_elements_flange.setMinimum(1)
        self.spinBox_number_elements_flange.setSingleStep(1)
        self.spinBox_number_elements_flange.setProperty("value", 1)
        self.spinBox_number_elements_flange.setObjectName("spinBox_number_elements_flange")
        self.lineEdit_flange_length = QtWidgets.QLineEdit(self.tab_flange_setup)
        self.lineEdit_flange_length.setEnabled(False)
        self.lineEdit_flange_length.setGeometry(QtCore.QRect(190, 84, 120, 30))
        self.lineEdit_flange_length.setMinimumSize(QtCore.QSize(120, 30))
        self.lineEdit_flange_length.setMaximumSize(QtCore.QSize(120, 30))
        self.lineEdit_flange_length.setSizeIncrement(QtCore.QSize(0, 26))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.lineEdit_flange_length.setFont(font)
        self.lineEdit_flange_length.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_flange_length.setStyleSheet("color: rgb(0, 0, 255);\n"
"background-color: rgb(255, 255, 255);")
        self.lineEdit_flange_length.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_flange_length.setObjectName("lineEdit_flange_length")
        self.label_118 = QtWidgets.QLabel(self.tab_flange_setup)
        self.label_118.setEnabled(True)
        self.label_118.setGeometry(QtCore.QRect(314, 84, 40, 30))
        self.label_118.setMinimumSize(QtCore.QSize(40, 30))
        self.label_118.setMaximumSize(QtCore.QSize(40, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_118.setFont(font)
        self.label_118.setMouseTracking(True)
        self.label_118.setAlignment(QtCore.Qt.AlignCenter)
        self.label_118.setObjectName("label_118")
        self.tabWidget_inputs.addTab(self.tab_flange_setup, "")
        self.checkBox_add_flanges_to_the_valve = QtWidgets.QCheckBox(self.frame)
        self.checkBox_add_flanges_to_the_valve.setGeometry(QtCore.QRect(146, 60, 250, 27))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_add_flanges_to_the_valve.setFont(font)
        self.checkBox_add_flanges_to_the_valve.setIconSize(QtCore.QSize(16, 16))
        self.checkBox_add_flanges_to_the_valve.setChecked(True)
        self.checkBox_add_flanges_to_the_valve.setTristate(False)
        self.checkBox_add_flanges_to_the_valve.setObjectName("checkBox_add_flanges_to_the_valve")
        self.checkBox_enable_acoustic_effects = QtWidgets.QCheckBox(self.frame)
        self.checkBox_enable_acoustic_effects.setGeometry(QtCore.QRect(146, 96, 250, 27))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.checkBox_enable_acoustic_effects.setFont(font)
        self.checkBox_enable_acoustic_effects.setIconSize(QtCore.QSize(16, 16))
        self.checkBox_enable_acoustic_effects.setChecked(True)
        self.checkBox_enable_acoustic_effects.setTristate(False)
        self.checkBox_enable_acoustic_effects.setObjectName("checkBox_enable_acoustic_effects")
        self.top_frame = QtWidgets.QFrame(Dialog)
        self.top_frame.setGeometry(QtCore.QRect(0, 0, 500, 59))
        self.top_frame.setMinimumSize(QtCore.QSize(500, 59))
        self.top_frame.setMaximumSize(QtCore.QSize(500, 59))
        self.top_frame.setFrameShape(QtWidgets.QFrame.Box)
        self.top_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.top_frame.setLineWidth(1)
        self.top_frame.setObjectName("top_frame")
        self.label = QtWidgets.QLabel(self.top_frame)
        self.label.setGeometry(QtCore.QRect(12, 8, 309, 43))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.radioButton_line_selection = QtWidgets.QRadioButton(self.top_frame)
        self.radioButton_line_selection.setGeometry(QtCore.QRect(336, 6, 155, 20))
        self.radioButton_line_selection.setMinimumSize(QtCore.QSize(155, 20))
        self.radioButton_line_selection.setMaximumSize(QtCore.QSize(130, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.radioButton_line_selection.setFont(font)
        self.radioButton_line_selection.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.radioButton_line_selection.setChecked(True)
        self.radioButton_line_selection.setObjectName("radioButton_line_selection")
        self.buttonGroup = QtWidgets.QButtonGroup(Dialog)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.radioButton_line_selection)
        self.radioButton_element_selection = QtWidgets.QRadioButton(self.top_frame)
        self.radioButton_element_selection.setGeometry(QtCore.QRect(336, 32, 155, 20))
        self.radioButton_element_selection.setMinimumSize(QtCore.QSize(155, 20))
        self.radioButton_element_selection.setMaximumSize(QtCore.QSize(155, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.radioButton_element_selection.setFont(font)
        self.radioButton_element_selection.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.radioButton_element_selection.setChecked(False)
        self.radioButton_element_selection.setObjectName("radioButton_element_selection")
        self.buttonGroup.addButton(self.radioButton_element_selection)

        self.retranslateUi(Dialog)
        self.tabWidget_inputs.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Add a valve"))
        self.pushButton_confirm.setText(_translate("Dialog", "Confirm"))
        self.label_selected_id.setText(_translate("Dialog", "<html><head/><body><p align=\"right\"><span style=\" font-style:normal;\">Selected ID:</span></p></body></html>"))
        self.label_selected_id_9.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">Valve length:</p></body></html>"))
        self.label_109.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-style:normal;\">[m]</span></p></body></html>"))
        self.label_selected_id_13.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">Valve mass:</p></body></html>"))
        self.label_108.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-style:normal;\">[kg]</span></p></body></html>"))
        self.label_112.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-style:normal;\">[ - ]</span></p></body></html>"))
        self.label_selected_id_14.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">Stiffening factor:</p></body></html>"))
        self.lineEdit_stiffening_factor_line.setText(_translate("Dialog", "10"))
        self.tabWidget_inputs.setTabText(self.tabWidget_inputs.indexOf(self.tab_insert_by_line), _translate("Dialog", "Line selection"))
        self.label_107.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-style:normal;\">[kg]</span></p></body></html>"))
        self.label_104.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-style:normal;\">[m]</span></p></body></html>"))
        self.label_selected_id_7.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">Valve mass:</p></body></html>"))
        self.label_selected_id_12.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">Valve length:</p></body></html>"))
        self.lineEdit_valve_length_element.setToolTip(_translate("Dialog", "Diametros"))
        self.label_111.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-style:normal;\">[ - ]</span></p></body></html>"))
        self.lineEdit_stiffening_factor_element.setText(_translate("Dialog", "10"))
        self.label_selected_id_8.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">Stiffening factor:</p></body></html>"))
        self.tabWidget_inputs.setTabText(self.tabWidget_inputs.indexOf(self.tab_insert_by_element), _translate("Dialog", "Element selection"))
        self.label_128.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-style:normal;\">[m]</span></p></body></html>"))
        self.label_outer_diameter.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">Outer diameter:</p></body></html>"))
        self.label_selected_id_23.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">Flange length:</p></body></html>"))
        self.label_selected_id_24.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">Number of elements:</p></body></html>"))
        self.label_118.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-style:normal;\">[m]</span></p></body></html>"))
        self.tabWidget_inputs.setTabText(self.tabWidget_inputs.indexOf(self.tab_flange_setup), _translate("Dialog", "Flange setup"))
        self.checkBox_add_flanges_to_the_valve.setText(_translate("Dialog", "Add flanges to the valve"))
        self.checkBox_enable_acoustic_effects.setText(_translate("Dialog", "Enable acoustic effects"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p align=\"center\">Add a valve at the pipeline by:</p></body></html>"))
        self.radioButton_line_selection.setText(_translate("Dialog", "Line selection"))
        self.radioButton_element_selection.setText(_translate("Dialog", "Element selection"))
