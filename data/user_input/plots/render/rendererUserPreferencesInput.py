from PyQt5.QtWidgets import QToolButton, QLineEdit, QDialog, QTabWidget, QLabel, QCheckBox, QRadioButton, QWidget
from data.user_input.project.printMessageInput import PrintMessageInput
from PyQt5.QtGui import QIcon
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5 import uic
import numpy as np

from pulse.interface.opvRenderer import PlotFilter, SelectionFilter

class RendererUserPreferencesInput(QDialog):
    def __init__(self, project, opv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi('data/user_input/ui/Plots/Render/rendererUserPreferencesInput.ui', self)
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowModality(Qt.WindowModal)

        icons_path = 'data\\icons\\'
        self.icon = QIcon(icons_path + 'pulse.png')
        self.setWindowIcon(self.icon)

        self.project = project
        self.opv = opv
        self.opv.setInputObject(self)

        self.checkBox_nodes_viewer = self.findChild(QCheckBox, 'checkBox_nodes_viewer')
        self.checkBox_elements_viewer = self.findChild(QCheckBox, 'checkBox_elements_viewer')
        self.checkBox_acoustic_symbols_viewer = self.findChild(QCheckBox, 'checkBox_acoustic_symbols_viewer')
        self.checkBox_structural_symbols_viewer = self.findChild(QCheckBox, 'checkBox_structural_symbols_viewer')

        self.checkBox_nodes_selector = self.findChild(QCheckBox, 'checkBox_nodes_selector')
        self.checkBox_elements_selector = self.findChild(QCheckBox, 'checkBox_elements_selector')
        self.checkBox_lines_selector = self.findChild(QCheckBox, 'checkBox_lines_selector')

        self.radioButton_black_color = self.findChild(QRadioButton, 'radioButton_black_color')
        self.radioButton_dark_gray_color = self.findChild(QRadioButton, 'radioButton_dark_gray_color')
        self.radioButton_light_gray_color = self.findChild(QRadioButton, 'radioButton_light_gray_color')
        self.radioButton_white_color = self.findChild(QRadioButton, 'radioButton_white_color')

        self.checkBox_OpenPulse_logo = self.findChild(QCheckBox, 'checkBox_OpenPulse_logo')
        self.checkBox_MOPT_logo = self.findChild(QCheckBox, 'checkBox_MOPT_logo')
        self.checkBox_reference_scale = self.findChild(QCheckBox, 'checkBox_reference_scale')

        self.tabWidget_main = self.findChild(QTabWidget, 'tabWidget_main')
        self.tab_hide_show = self.tabWidget_main.findChild(QWidget, 'tab_hide_show')
        self.tab_user_preferences = self.tabWidget_main.findChild(QWidget, 'tab_user_preferences')
        self.tabWidget_main.removeTab(0)
        
        self.toolButton_confirm = self.findChild(QToolButton, 'toolButton_confirm')
        self.toolButton_confirm.clicked.connect(self.confirm_and_update_user_preferences)
        self.load_background_color_state()
        self.load_logo_state()
        self.load_reference_scale_state()
        # self.load_plot_state()
        # self.load_selection_state()
        self.exec()

    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            self.confirm_and_update_user_preferences()
        elif event.key() == Qt.Key_Escape:
            self.close()

    def load_plot_state(self):
        '''
        Check over the renderer wich components are being plotted to update the checkbox states.
        '''

        plot_filter = self.opv.opvRenderer._plotFilter

        self.checkBox_nodes_viewer.setChecked(
            PlotFilter.nodes & plot_filter
        )
        self.checkBox_elements_viewer.setChecked(
            PlotFilter.tubes & plot_filter
        )
        self.checkBox_acoustic_symbols_viewer.setChecked(
            PlotFilter.acoustic_symbols & plot_filter
        )
        self.checkBox_structural_symbols_viewer.setChecked(
            PlotFilter.structural_symbols & plot_filter
        )

    def load_selection_state(self):
        '''
        Check over the renderer wich components are being selected to update the checkbox states.
        '''

        selection_filter = self.opv.opvRenderer._selectionFilter

        self.checkBox_nodes_selector.setChecked(
            SelectionFilter.nodes & selection_filter
        )
        self.checkBox_elements_selector.setChecked(
            SelectionFilter.elements & selection_filter
        )
        self.checkBox_lines_selector.setChecked(
            SelectionFilter.entities & selection_filter
        )
    
    def update_selection_state(self):
        '''
        Reads the users options and updates selection behavior.
        '''

        plt_nodes = self.checkBox_nodes_viewer.isChecked()
        slc_nodes = self.checkBox_nodes_selector.isChecked()
        slc_elements = self.checkBox_elements_selector.isChecked()
        slc_entities = self.checkBox_lines_selector.isChecked()

        self.opv.opvRenderer.setSelectionFilter(
            (SelectionFilter.nodes if slc_nodes and plt_nodes else 0)
            | (SelectionFilter.elements if slc_elements else 0)
            | (SelectionFilter.entities if slc_entities else 0)
        )
    
    def update_plot_state(self):
        '''
        Reads the users options and updates the plot.
        '''

        plt_nodes = self.checkBox_nodes_viewer.isChecked()
        plt_tubes = self.checkBox_elements_viewer.isChecked()
        plt_acoustic = self.checkBox_acoustic_symbols_viewer.isChecked()
        plt_structural = self.checkBox_structural_symbols_viewer.isChecked()

        self.opv.opvRenderer.setPlotFilter(
            (PlotFilter.lines)
            | (PlotFilter.nodes if plt_nodes else 0)
            | (PlotFilter.tubes if plt_tubes else 0)
            | (PlotFilter.acoustic_symbols if plt_acoustic else 0)
            | (PlotFilter.structural_symbols if plt_structural else 0)
            | (PlotFilter.transparent if (plt_nodes or plt_acoustic or plt_structural) else 0)
        )

    def load_background_color_state(self):
        if self.opv.background_color == (0,0,0):
            self.radioButton_black_color.setChecked(True)
        elif self.opv.background_color == (0.25,0.25,0.25):
            self.radioButton_dark_gray_color.setChecked(True)
        elif self.opv.background_color == (0.7,0.7,0.7):
            self.radioButton_light_gray_color.setChecked(True)
        elif self.opv.background_color == (1,1,1):
            self.radioButton_white_color.setChecked(True)

    def update_background_color_state(self):

        if self.radioButton_black_color.isChecked():
            color = (0,0,0)
            font_color = (1,1,1)
        elif self.radioButton_dark_gray_color.isChecked():
            color = (0.25,0.25,0.25)
            font_color = (1,1,1)
        elif self.radioButton_light_gray_color.isChecked():
            color = (0.7,0.7,0.7)
            font_color = (0,0,0)
        elif self.radioButton_white_color.isChecked():
            color = (1,1,1)
            font_color = (0,0,0)
        self.opv.background_color = color
        self.opv.font_color = font_color

        self.opv.opvRenderer.changeBackgroundColor(color)
        self.opv.opvAnalysisRenderer.changeBackgroundColor(color)
        self.opv.opvRenderer.changeFontColor(font_color)
        self.opv.opvAnalysisRenderer.changeFontColor(font_color)
        self.opv.opvRenderer._updateFontColor(font_color)
        self.opv.opvAnalysisRenderer._updateFontColor(font_color)
    
    def load_reference_scale_state(self):
        self.checkBox_reference_scale.setChecked(self.opv.show_reference_scale)

    def update_reference_scale_state(self):
        self.opv.show_reference_scale = self.checkBox_reference_scale.isChecked()
        self.opv.opvRenderer._createScaleBar()
        self.opv.opvAnalysisRenderer._createScaleBar()

    def load_logo_state(self):
        self.checkBox_OpenPulse_logo.setChecked(self.opv.add_OpenPulse_logo)
        self.checkBox_MOPT_logo.setChecked(self.opv.add_MOPT_logo)
            
    def update_logo_state(self):        
        self.opv.add_OpenPulse_logo = self.checkBox_OpenPulse_logo.isChecked()
        self.opv.add_MOPT_logo = self.checkBox_MOPT_logo.isChecked()
        self.opv.opvRenderer._createLogos(OpenPulse=self.opv.add_OpenPulse_logo, MOPT=self.opv.add_MOPT_logo)
        self.opv.opvAnalysisRenderer._createLogos(OpenPulse=self.opv.add_OpenPulse_logo, MOPT=self.opv.add_MOPT_logo)

    def confirm_and_update_user_preferences(self):
        self.update_plot_state()
        self.update_selection_state()
        self.update_logo_state()
        self.update_background_color_state()
        self.update_reference_scale_state()
        preferences = { 'background-color' : self.opv.background_color,
                        'font-color' : self.opv.font_color,
                        'OpenPulse logo' : int(self.opv.add_OpenPulse_logo),
                        'mopt logo' : int(self.opv.add_MOPT_logo),
                        'Reference scale' : int(self.opv.show_reference_scale) }
        self.project.add_user_preferences_to_file(preferences)
        # self.update_renders()
        self.close()

    def update_renders(self):
        self.opv.updateRendererMesh()
        if self.opv.change_plot_to_mesh:
            self.opv.changePlotToMesh()
        elif self.opv.change_plot_to_entities:
            self.opv.changePlotToEntities()
        elif self.opv.change_plot_to_entities_with_cross_section:
            self.opv.changePlotToEntitiesWithCrossSection()
