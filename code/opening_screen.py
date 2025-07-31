import sys
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QApplication, QMessageBox

from opening_screen_ui import Ui_OpeningScreen
from analysis_pipeline_screen import AnalysisPipelineScreen

class OpeningScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_OpeningScreen() # Initialize the UI designed in Qt Designer
        self.ui.setupUi(self)        # Set up UI elements in this window

        self.opening_screen = OpeningScreen # Save the reference
        self.selected_folder_path = None  # Store selected folder

        # Connect buttons
        self.ui.startButton.clicked.connect(self.go_to_analysis_pipeline)
        self.ui.aboutButton.clicked.connect(self.show_about_dialog)
        self.ui.userGuideButton.clicked.connect(self.show_user_guide_dialog)

    def go_to_analysis_pipeline(self):
        """Transition from the opening screen to the analysis pipeline screen.
           Passes the selected folder path to the next screen."""
        self.analysis_window = AnalysisPipelineScreen(opening_screen=self, preselected_folder=self.selected_folder_path)
        self.analysis_window.show()
        self.hide()

    def show_about_dialog(self):
        """Display a custom About dialog with project information."""
        # Create a custom QDialog
        about_dialog = QtWidgets.QDialog(self)
        about_dialog.setWindowTitle("About")
        about_dialog.setFixedSize(500, 370)

        # Set custom background color here
        about_dialog.setStyleSheet("background-color: #F1F5F9;")  # Light gray-blue

        # Create and style layout and label
        layout = QtWidgets.QVBoxLayout(about_dialog)

        # Create and configure label with about text
        label = QtWidgets.QLabel(
            "Welcome to the fNIRS Hyperscanning Analysis Tool!\n\n"
            "This tool allows neuroscientists to analyze fNIRS hyperscanning data using graph theory measures.\n\n"
            "You can generate correlation matrices, apply thresholding, and compare global and local graph metrics "
            "between conditions and dyads.\n\n"
            "Developed as a capstone project for B.Sc. in\nSoftware Engineering by Yonatan Sherer, 2025."
        )
        label.setWordWrap(True)
        label.setStyleSheet("font-size: 20px; color: #1F2937;")  # Optional text style

        layout.addWidget(label)

        # Add a Close button
        close_button = QtWidgets.QPushButton("Close")
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #00FFFF;
                border-radius: 8px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #00CCCC;
            }
            QPushButton:pressed {
                background-color: #009999;
            }
        """)
        close_button.clicked.connect(about_dialog.accept)
        layout.addWidget(close_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        about_dialog.exec()

    def show_user_guide_dialog(self):
        # Create a custom QDialog
        about_dialog = QtWidgets.QDialog(self)
        about_dialog.setWindowTitle("User Guide")
        about_dialog.setFixedSize(650, 700)

        # Set custom background color here
        about_dialog.setStyleSheet("background-color: #F1F5F9;")  # Light gray-blue

        # Create and style layout and label
        layout = QtWidgets.QVBoxLayout(about_dialog)

        label = QtWidgets.QLabel(
            "🧠 Welcome to the fNIRS Hyperscanning Tool!\n\n"
    "① Select Folder\n"
    "Choose your experiment folder to begin.\n\n"
    "② Organize Files\n"
    "Sort raw .mat files by group and condition.\n\n"
    "③ Generate Correlations\n"
    "Compute correlation matrices and heatmaps.\n\n"
    "④ Apply Thresholding\n"
    "Select a method to refine matrices.\n\n"
    "⑤ Analyze Graphs\n"
    "Extract global & local graph metrics.\n\n"
    "⑥ Compare Measures\n"
    "Visualize and compare metrics across dyads, conditions, and nodes.\n\n"
    "📤 Export results anytime as CSV, images, and PDF.\n\n"
    "ℹ️ Use tooltips and Help buttons on each screen for more info.\n"
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet("font-size: 20px; color: #1F2937;")  # Optional text style

        layout.addWidget(label)

        # Add a Close button
        close_button = QtWidgets.QPushButton("Close")
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #00FFFF;
                border-radius: 8px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #00CCCC;
            }
            QPushButton:pressed {
                background-color: #009999;
            }
        """)
        close_button.clicked.connect(about_dialog.accept)
        layout.addWidget(close_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        about_dialog.exec() # Display the dialog

# Launch the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OpeningScreen()
    window.show()
    sys.exit(app.exec()) # Run the event loop