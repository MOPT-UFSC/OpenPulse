# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Kula\Petrobras\OpenPulse\data\user_input\ui\Plots\Render\meshSetupVisibilityInput.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(360, 500)
        Dialog.setMinimumSize(QtCore.QSize(360, 500))
        Dialog.setMaximumSize(QtCore.QSize(360, 500))
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setGeometry(QtCore.QRect(0, 0, 360, 500))
        self.frame.setMinimumSize(QtCore.QSize(360, 270))
        self.frame.setMaximumSize(QtCore.QSize(360, 500))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setObjectName("frame")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(0, 0, 360, 40))
        self.frame_2.setMinimumSize(QtCore.QSize(360, 40))
        self.frame_2.setMaximumSize(QtCore.QSize(360, 40))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_2.setLineWidth(1)
        self.frame_2.setObjectName("frame_2")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setGeometry(QtCore.QRect(28, 6, 300, 28))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.toolButton_confirm = QtWidgets.QToolButton(self.frame)
        self.toolButton_confirm.setGeometry(QtCore.QRect(108, 460, 140, 30))
        self.toolButton_confirm.setMinimumSize(QtCore.QSize(140, 30))
        self.toolButton_confirm.setMaximumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.toolButton_confirm.setFont(font)
        self.toolButton_confirm.setObjectName("toolButton_confirm")
        self.tabWidget = QtWidgets.QTabWidget(self.frame)
        self.tabWidget.setGeometry(QtCore.QRect(28, 52, 300, 400))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.frame_4 = QtWidgets.QFrame(self.tab)
        self.frame_4.setGeometry(QtCore.QRect(28, 240, 240, 120))
        self.frame_4.setMinimumSize(QtCore.QSize(240, 120))
        self.frame_4.setMaximumSize(QtCore.QSize(240, 120))
        self.frame_4.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.checkBox_elements_selector = QtWidgets.QCheckBox(self.frame_4)
        self.checkBox_elements_selector.setGeometry(QtCore.QRect(34, 50, 170, 22))
        self.checkBox_elements_selector.setMinimumSize(QtCore.QSize(170, 22))
        self.checkBox_elements_selector.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.checkBox_elements_selector.setFont(font)
        self.checkBox_elements_selector.setChecked(True)
        self.checkBox_elements_selector.setObjectName("checkBox_elements_selector")
        self.checkBox_nodes_selector = QtWidgets.QCheckBox(self.frame_4)
        self.checkBox_nodes_selector.setGeometry(QtCore.QRect(34, 14, 170, 22))
        self.checkBox_nodes_selector.setMinimumSize(QtCore.QSize(170, 22))
        self.checkBox_nodes_selector.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.checkBox_nodes_selector.setFont(font)
        self.checkBox_nodes_selector.setChecked(True)
        self.checkBox_nodes_selector.setObjectName("checkBox_nodes_selector")
        self.checkBox_lines_selector = QtWidgets.QCheckBox(self.frame_4)
        self.checkBox_lines_selector.setGeometry(QtCore.QRect(34, 86, 170, 22))
        self.checkBox_lines_selector.setMinimumSize(QtCore.QSize(170, 22))
        self.checkBox_lines_selector.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.checkBox_lines_selector.setFont(font)
        self.checkBox_lines_selector.setChecked(True)
        self.checkBox_lines_selector.setObjectName("checkBox_lines_selector")
        self.frame_3 = QtWidgets.QFrame(self.tab)
        self.frame_3.setGeometry(QtCore.QRect(28, 40, 240, 160))
        self.frame_3.setMinimumSize(QtCore.QSize(240, 160))
        self.frame_3.setMaximumSize(QtCore.QSize(240, 160))
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.checkBox_structural_symbols_viewer = QtWidgets.QCheckBox(self.frame_3)
        self.checkBox_structural_symbols_viewer.setGeometry(QtCore.QRect(34, 122, 170, 22))
        self.checkBox_structural_symbols_viewer.setMinimumSize(QtCore.QSize(170, 22))
        self.checkBox_structural_symbols_viewer.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.checkBox_structural_symbols_viewer.setFont(font)
        self.checkBox_structural_symbols_viewer.setChecked(True)
        self.checkBox_structural_symbols_viewer.setObjectName("checkBox_structural_symbols_viewer")
        self.checkBox_elements_viewer = QtWidgets.QCheckBox(self.frame_3)
        self.checkBox_elements_viewer.setGeometry(QtCore.QRect(34, 50, 170, 22))
        self.checkBox_elements_viewer.setMinimumSize(QtCore.QSize(170, 22))
        self.checkBox_elements_viewer.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.checkBox_elements_viewer.setFont(font)
        self.checkBox_elements_viewer.setChecked(True)
        self.checkBox_elements_viewer.setObjectName("checkBox_elements_viewer")
        self.checkBox_nodes_viewer = QtWidgets.QCheckBox(self.frame_3)
        self.checkBox_nodes_viewer.setGeometry(QtCore.QRect(34, 14, 170, 22))
        self.checkBox_nodes_viewer.setMinimumSize(QtCore.QSize(170, 22))
        self.checkBox_nodes_viewer.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.checkBox_nodes_viewer.setFont(font)
        self.checkBox_nodes_viewer.setChecked(True)
        self.checkBox_nodes_viewer.setObjectName("checkBox_nodes_viewer")
        self.checkBox_acoustic_symbols_viewer = QtWidgets.QCheckBox(self.frame_3)
        self.checkBox_acoustic_symbols_viewer.setGeometry(QtCore.QRect(34, 86, 170, 22))
        self.checkBox_acoustic_symbols_viewer.setMinimumSize(QtCore.QSize(170, 22))
        self.checkBox_acoustic_symbols_viewer.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.checkBox_acoustic_symbols_viewer.setFont(font)
        self.checkBox_acoustic_symbols_viewer.setChecked(True)
        self.checkBox_acoustic_symbols_viewer.setObjectName("checkBox_acoustic_symbols_viewer")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(28, 210, 240, 32))
        self.label_3.setMinimumSize(QtCore.QSize(240, 32))
        self.label_3.setMaximumSize(QtCore.QSize(240, 32))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(28, 10, 240, 32))
        self.label_2.setMinimumSize(QtCore.QSize(240, 32))
        self.label_2.setMaximumSize(QtCore.QSize(240, 32))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.frame_6 = QtWidgets.QFrame(self.tab_2)
        self.frame_6.setGeometry(QtCore.QRect(28, 240, 240, 120))
        self.frame_6.setMinimumSize(QtCore.QSize(240, 120))
        self.frame_6.setMaximumSize(QtCore.QSize(240, 120))
        self.frame_6.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.checkBox_MOPT_logo = QtWidgets.QCheckBox(self.frame_6)
        self.checkBox_MOPT_logo.setGeometry(QtCore.QRect(34, 50, 170, 22))
        self.checkBox_MOPT_logo.setMinimumSize(QtCore.QSize(170, 22))
        self.checkBox_MOPT_logo.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.checkBox_MOPT_logo.setFont(font)
        self.checkBox_MOPT_logo.setChecked(True)
        self.checkBox_MOPT_logo.setObjectName("checkBox_MOPT_logo")
        self.checkBox_OpenPulse_logo = QtWidgets.QCheckBox(self.frame_6)
        self.checkBox_OpenPulse_logo.setGeometry(QtCore.QRect(34, 14, 170, 22))
        self.checkBox_OpenPulse_logo.setMinimumSize(QtCore.QSize(170, 22))
        self.checkBox_OpenPulse_logo.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.checkBox_OpenPulse_logo.setFont(font)
        self.checkBox_OpenPulse_logo.setChecked(True)
        self.checkBox_OpenPulse_logo.setObjectName("checkBox_OpenPulse_logo")
        self.checkBox_reference_scale = QtWidgets.QCheckBox(self.frame_6)
        self.checkBox_reference_scale.setGeometry(QtCore.QRect(34, 86, 170, 22))
        self.checkBox_reference_scale.setMinimumSize(QtCore.QSize(170, 22))
        self.checkBox_reference_scale.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.checkBox_reference_scale.setFont(font)
        self.checkBox_reference_scale.setChecked(True)
        self.checkBox_reference_scale.setObjectName("checkBox_reference_scale")
        self.frame_5 = QtWidgets.QFrame(self.tab_2)
        self.frame_5.setGeometry(QtCore.QRect(28, 40, 240, 160))
        self.frame_5.setMinimumSize(QtCore.QSize(240, 160))
        self.frame_5.setMaximumSize(QtCore.QSize(240, 160))
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.radioButton_light_gray_color = QtWidgets.QRadioButton(self.frame_5)
        self.radioButton_light_gray_color.setGeometry(QtCore.QRect(34, 86, 170, 22))
        self.radioButton_light_gray_color.setMinimumSize(QtCore.QSize(170, 22))
        self.radioButton_light_gray_color.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_light_gray_color.setFont(font)
        self.radioButton_light_gray_color.setChecked(False)
        self.radioButton_light_gray_color.setObjectName("radioButton_light_gray_color")
        self.radioButton_black_color = QtWidgets.QRadioButton(self.frame_5)
        self.radioButton_black_color.setGeometry(QtCore.QRect(34, 14, 170, 22))
        self.radioButton_black_color.setMinimumSize(QtCore.QSize(170, 22))
        self.radioButton_black_color.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_black_color.setFont(font)
        self.radioButton_black_color.setChecked(True)
        self.radioButton_black_color.setObjectName("radioButton_black_color")
        self.radioButton_white_color = QtWidgets.QRadioButton(self.frame_5)
        self.radioButton_white_color.setGeometry(QtCore.QRect(34, 122, 170, 22))
        self.radioButton_white_color.setMinimumSize(QtCore.QSize(170, 22))
        self.radioButton_white_color.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_white_color.setFont(font)
        self.radioButton_white_color.setChecked(False)
        self.radioButton_white_color.setObjectName("radioButton_white_color")
        self.radioButton_dark_gray_color = QtWidgets.QRadioButton(self.frame_5)
        self.radioButton_dark_gray_color.setGeometry(QtCore.QRect(34, 50, 170, 22))
        self.radioButton_dark_gray_color.setMinimumSize(QtCore.QSize(170, 22))
        self.radioButton_dark_gray_color.setMaximumSize(QtCore.QSize(170, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_dark_gray_color.setFont(font)
        self.radioButton_dark_gray_color.setChecked(False)
        self.radioButton_dark_gray_color.setObjectName("radioButton_dark_gray_color")
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setGeometry(QtCore.QRect(28, 210, 240, 32))
        self.label_5.setMinimumSize(QtCore.QSize(240, 32))
        self.label_5.setMaximumSize(QtCore.QSize(240, 32))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setFrameShape(QtWidgets.QFrame.Box)
        self.label_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.label_4.setGeometry(QtCore.QRect(28, 10, 240, 32))
        self.label_4.setMinimumSize(QtCore.QSize(240, 32))
        self.label_4.setMaximumSize(QtCore.QSize(240, 32))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.tabWidget.addTab(self.tab_2, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Mesh setup visibility"))
        self.label.setText(_translate("Dialog", "User interface visibility settings"))
        self.toolButton_confirm.setText(_translate("Dialog", "Confirm"))
        self.checkBox_elements_selector.setText(_translate("Dialog", "Elements"))
        self.checkBox_nodes_selector.setText(_translate("Dialog", " Nodes"))
        self.checkBox_lines_selector.setText(_translate("Dialog", "Lines"))
        self.checkBox_structural_symbols_viewer.setText(_translate("Dialog", "Structural symbols"))
        self.checkBox_elements_viewer.setText(_translate("Dialog", "Elements"))
        self.checkBox_nodes_viewer.setText(_translate("Dialog", "Nodes"))
        self.checkBox_acoustic_symbols_viewer.setText(_translate("Dialog", "Acoustic symbols"))
        self.label_3.setText(_translate("Dialog", "Enable/Disable selection"))
        self.label_2.setText(_translate("Dialog", "Show/Hide"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "Model"))
        self.checkBox_MOPT_logo.setText(_translate("Dialog", "MOPT logo"))
        self.checkBox_OpenPulse_logo.setText(_translate("Dialog", "OpenPulse logo"))
        self.checkBox_reference_scale.setText(_translate("Dialog", "Reference scale"))
        self.radioButton_light_gray_color.setText(_translate("Dialog", "Light gray"))
        self.radioButton_black_color.setText(_translate("Dialog", "Black"))
        self.radioButton_white_color.setText(_translate("Dialog", "White"))
        self.radioButton_dark_gray_color.setText(_translate("Dialog", "Dark gray"))
        self.label_5.setText(_translate("Dialog", "Show/Hide"))
        self.label_4.setText(_translate("Dialog", "Background color"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "Advanced"))
