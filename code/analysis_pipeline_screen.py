import os
import shutil
import re
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat, savemat
import networkx as nx
import pandas as pd

from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QApplication, QLabel, QPushButton, QComboBox, QProgressBar, QProgressDialog
)
from PyQt6.QtGui import QCursor
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6 import uic, QtWidgets, QtCore
from PyQt6.QtWidgets import QInputDialog
from PyQt6.QtCore import QDateTime

from graph_measures_comparison_screen import GraphMeasureComparisonWindow
from local_measures_comparison_screen import LocalMeasureComparisonWindow

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet("color: blue; text-decoration: underline;")

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

class AnalysisPipelineScreen(QMainWindow):
    def __init__(self, opening_screen=None, preselected_folder=None):
        super().__init__()

        from analysis_pipeline_ui import Ui_AnalysisPipelineScreen

        self.ui = Ui_AnalysisPipelineScreen()
        self.ui.setupUi(self)

        self.selected_folder_path = None
        self.opening_screen = opening_screen
        self.preselected_folder = preselected_folder

        # Replace QLabel with a clickable version
        old_label = self.findChild(QLabel, "folderPathLabel")
        self.folderPathLabel = ClickableLabel(self)
        self.folderPathLabel.setGeometry(old_label.geometry())
        self.folderPathLabel.setObjectName("folderPathLabel")  # Preserve the name if needed
        self.folderPathLabel.show()
        old_label.hide()

        # Set folder path if passed in
        if self.preselected_folder:
            self.selected_folder_path = self.preselected_folder
            self.folderPathLabel.setText(self.selected_folder_path)
        else:
            self.folderPathLabel.setText("No folder selected")

        # Connect buttons
        self.selectFolderButton = self.findChild(QPushButton, "selectFolderButton")
        self.selectFolderButton.clicked.connect(self.select_folder)

        self.organizeFilesButton = self.findChild(QPushButton, "organizeFilesButton")
        self.organizeFilesButton.clicked.connect(self.organize_files)

        self.generateCorrelationsButton = self.findChild(QPushButton, "generateCorrelationsButton")
        self.generateCorrelationsButton.clicked.connect(self.generate_correlations)

        self.thresholdButton = self.findChild(QPushButton, "thresholdButton")
        self.thresholdButton.clicked.connect(self.run_thresholding)
        self.thresholdMethodComboBox = self.findChild(QComboBox, "thresholdMethodComboBox")

        self.folderPathLabel.clicked.connect(self.open_selected_folder)

        self.correlationProgressBar = self.findChild(QProgressBar, "correlationProgressBar")
        self.correlationProgressBar.setVisible(False)

        self.analyzeGraphsButton = self.findChild(QPushButton, "analyzeGraphsButton")
        self.analyzeGraphsButton.clicked.connect(self.analyze_hyperscanning_graphs)

        # Connect dropdown-based comparison logic
        self.comparisonModeComboBox = self.findChild(QComboBox, "comparisonModeComboBox")
        self.goButton = self.findChild(QPushButton, "goButton")

        if self.goButton and self.comparisonModeComboBox:
            self.goButton.clicked.connect(self.open_selected_comparison_screen)

        self.thresholdStatusLabel = self.findChild(QLabel, "thresholdStatusLabel")

        self.ui.helpButton.clicked.connect(self.show_help_dialog)
        self.ui.backButton.clicked.connect(self.go_back_to_opening_screen)

        #Tooltips
        self.selectFolderButton.setToolTip("Select folder containing experiment data")
        self.organizeFilesButton.setToolTip("Organize .mat files by condition")
        self.generateCorrelationsButton.setToolTip("Generate correlation matrices and heatmaps")
        self.thresholdButton.setToolTip("Apply thresholding method to correlation matrices")

        self.update_button_highlights()

    def select_folder(self):
        """Open a dialog for selecting the experiment root folder.
           Update the UI label to show the selected folder."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder_path:
            self.selected_folder_path = folder_path
            self.folderPathLabel.setText(folder_path)
            self.update_button_highlights()

    def open_selected_folder(self):
        if self.selected_folder_path and os.path.exists(self.selected_folder_path):
            try:
                webbrowser.open(self.selected_folder_path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open folder: {e}")
        else:
            QMessageBox.warning(self, "No Folder", "No valid folder selected.")

    def organize_files(self):
        """Trigger the file organization utility function.
           Sorts raw .mat files into a structure by group and condition."""
        if not self.selected_folder_path:
            self.log("No folder selected. Please choose a folder before organizing files.")
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder first.")
            return

        self.log(f"Starting file organization in: {self.selected_folder_path}")
        try:
            self.organize_mat_files_by_condition(self.selected_folder_path)
            self.log("Files organized successfully.")
            QMessageBox.information(self, "Success", "Files organized successfully.")
            self.update_button_highlights()
        except Exception as e:
            self.log(f"Error organizing files: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred:\n{e}")

    def organize_mat_files_by_condition(self, source_folder):
        experiment_folder = os.path.join(source_folder, "Experiment")
        os.makedirs(experiment_folder, exist_ok=True)

        pattern = re.compile(r"^dyad(\d+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)\.mat$")

        for filename in os.listdir(source_folder):
            if filename.endswith(".mat"):
                match = pattern.match(filename)
                if not match:
                    print(f"Skipping invalid file: {filename}")
                    continue

                dyad_id, group_name, condition_name, role = match.groups()

                group_folder = os.path.join(experiment_folder, group_name)
                condition_folder = os.path.join(group_folder, condition_name)
                os.makedirs(condition_folder, exist_ok=True)

                src_path = os.path.join(source_folder, filename)
                dst_path = os.path.join(condition_folder, filename)
                shutil.copy2(src_path, dst_path)

                print(f"Copied {filename} → {condition_folder}")

        print("\n✅ Organization complete.")

    from PyQt6.QtWidgets import QProgressDialog

    def generate_correlations(self):
        """Generate correlation matrices and save heatmaps for all dyads."""
        if not self.selected_folder_path:
            self.log("No folder selected. Please choose a folder before generating correlations.")
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder first.")
            return

        experiment_path = os.path.join(self.selected_folder_path, "Experiment")
        if not os.path.isdir(experiment_path):
            self.log(f"Missing Experiment folder at: {experiment_path}")
            QMessageBox.critical(self, "Missing Experiment Folder",
                                 f"'Experiment' folder not found in: {experiment_path}")
            return

        # Collect all condition folders from all group subfolders
        condition_paths = []
        for group_name in os.listdir(experiment_path):
            group_path = os.path.join(experiment_path, group_name)
            if not os.path.isdir(group_path):
                continue
            for condition_name in os.listdir(group_path):
                condition_path = os.path.join(group_path, condition_name)
                if os.path.isdir(condition_path):
                    condition_paths.append((group_name, condition_name, condition_path))

        if not condition_paths:
            self.log(f"No condition folders found under any group in {experiment_path}.")
            QMessageBox.information(self, "No Conditions", "No condition folders found to process.")
            return

        self.log(f"Starting correlation generation for {len(condition_paths)} condition(s)...")

        progress_dialog = QProgressDialog("Generating correlations...", "Cancel", 0, len(condition_paths), self)
        progress_dialog.setWindowTitle("Progress")
        progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)
        progress_dialog.show()

        for i, (group_name, condition_name, condition_path) in enumerate(condition_paths, start=1):
            if progress_dialog.wasCanceled():
                self.log("❌ Correlation generation canceled by user.")
                QMessageBox.information(self, "Cancelled", "⛔ Operation cancelled by user.")
                return

            self.log(f"Processing: Group={group_name}, Condition={condition_name} at {condition_path}")
            try:
                self.process_condition_folder(condition_path, condition_name)
                self.log(f"✅ Finished processing: {group_name}/{condition_name}")
            except Exception as e:
                self.log(f"❌ Error processing {group_name}/{condition_name}: {e}")

            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"Processing: {group_name}/{condition_name}")
            QApplication.processEvents()

        progress_dialog.setValue(len(condition_paths))
        self.log("✅ All conditions processed. Correlation matrices and heatmaps saved.")
        QMessageBox.information(self, "Done", "✅ Correlation .mat files and heatmaps saved.")
        self.update_button_highlights()

    def process_condition_folder(self, condition_path, condition_name):
        files = [f for f in os.listdir(condition_path) if f.endswith(".mat")]
        dyad_dict = {}

        for file in files:
            match = re.match(r"(dyad\d+)_([^_]+)_(.+)\.mat", file)
            if not match:
                continue
            dyad, _, participant = match.groups()  # We ignore the condition from filename now
            dyad_dict.setdefault(dyad, {})[participant] = os.path.join(condition_path, file)

        for dyad, pair in dyad_dict.items():
            if len(pair) != 2:
                print(f"Skipping {dyad} – incomplete pair")
                continue

            p1, p2 = list(pair.keys())
            file1, file2 = pair[p1], pair[p2]

            try:
                data1 = self.extract_data(file1)
                data2 = self.extract_data(file2)
            except Exception as e:
                print(f"Error reading {file1} or {file2}: {e}")
                continue

            if data1.shape[0] < data1.shape[1]:
                data1 = data1.T
            if data2.shape[0] < data2.shape[1]:
                data2 = data2.T

            combined = np.hstack([data1, data2])
            corr_matrix = np.corrcoef(combined.T)

            save_dir = os.path.join(condition_path, f"{condition_name}_correlations")
            os.makedirs(save_dir, exist_ok=True)

            mat_file_path = os.path.join(save_dir, f"{dyad}_{condition_name}_correlation.mat")
            savemat(mat_file_path, {"correlation": corr_matrix})

            heatmap_path = os.path.join(save_dir, f"{dyad}_{condition_name}_heatmap.png")
            self.plot_correlation_matrix(corr_matrix, heatmap_path, dyad, condition_name, data1.shape[1],
                                         data2.shape[1])

            print(f"Saved .mat and heatmap for {dyad} in {condition_name}.")

    def extract_data(self, mat_file_path):
        mat = loadmat(mat_file_path)
        for key in mat:
            if not key.startswith("__"):
                return mat[key]
        raise KeyError(f"No usable variable found in {mat_file_path}")

    def plot_correlation_matrix(self, corr_matrix, output_path, dyad, condition, channels_p1, channels_p2):
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm', square=True, cbar_kws={'label': 'Correlation'})
        plt.title(f"Correlation Matrix – {dyad} ({condition})")

        total_channels = channels_p1 + channels_p2
        ax.hlines([channels_p1], *ax.get_xlim(), colors='black', linewidth=1)
        ax.vlines([channels_p1], *ax.get_ylim(), colors='black', linewidth=1)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def run_thresholding(self):
        if not self.selected_folder_path:
            self.log("No folder selected. Please choose a folder before running thresholding.")
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder first.")
            return

        method = self.thresholdMethodComboBox.currentText()
        method_key = method.lower().replace(" ", "_").replace("-", "_")
        threshold_kwargs = {}

        if method_key == "fixed_threshold":
            value, ok = QInputDialog.getDouble(
                self,
                "Fixed Threshold",
                "Enter threshold value (between 0.3 and 1):",
                value=0.3,
                min=0.3,
                max=1.0,
                decimals=3
            )
            if not ok:
                self.log("Fixed threshold input cancelled by user.")
                return
            threshold_kwargs['method'] = 'fixed'
            threshold_kwargs['threshold'] = value
            self.log(f"Using Fixed Threshold: {value}")

        elif method_key == "median_based":
            threshold_kwargs['method'] = 'median'
            self.log("Using Median-Based Threshold")

        elif method_key == "top_percentile":
            value, ok = QInputDialog.getDouble(
                self,
                "Top Percentile",
                "Enter top percentage (e.g., 20 for top 20%):",
                min=1, max=100, decimals=1
            )
            if not ok:
                self.log("Top percentile input cancelled by user.")
                return
            threshold_kwargs['method'] = 'top_percent'
            threshold_kwargs['top_percent'] = value
            self.log(f"Using Top Percentile Threshold: {value}%")

        else:
            self.log(f"Unknown thresholding method selected: {method}")
            QMessageBox.warning(self, "Unknown Method", f"Unsupported method: {method}")
            return

        # Update threshold status label
        if method_key == "fixed_threshold":
            self.thresholdStatusLabel.setText(f"Current threshold: Fixed ({value:.3f})")
        elif method_key == "median_based":
            self.thresholdStatusLabel.setText("Current threshold: Median-based")
        elif method_key == "top_percentile":
            self.thresholdStatusLabel.setText(f"Current threshold: Top {value:.1f}%")
        else:
            self.thresholdStatusLabel.setText("Current threshold: Unknown method")

        try:
            experiment_path = os.path.join(self.selected_folder_path, "Experiment")
            all_conditions = []

            for group_name in os.listdir(experiment_path):
                group_path = os.path.join(experiment_path, group_name)
                if not os.path.isdir(group_path):
                    continue
                for condition_name in os.listdir(group_path):
                    condition_path = os.path.join(group_path, condition_name)
                    if os.path.isdir(condition_path):
                        all_conditions.append((group_name, condition_name, condition_path))

            total = len(all_conditions)
            if total == 0:
                self.log("No condition folders found in the experiment structure.")
                QMessageBox.warning(self, "No Conditions", "No condition folders found.")
                return

            self.log(f"Starting thresholding using method: {threshold_kwargs['method']} on {total} condition(s)...")

            progress_dialog = QProgressDialog("Applying threshold...", "Cancel", 0, total, self)
            progress_dialog.setWindowTitle("Thresholding Progress")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_dialog.setAutoClose(True)
            progress_dialog.setAutoReset(True)
            progress_dialog.show()

            for i, (group_name, condition_name, condition_path) in enumerate(all_conditions, start=1):
                if progress_dialog.wasCanceled():
                    self.log("❌ Thresholding cancelled by user.")
                    QMessageBox.information(self, "Cancelled", "⛔ Thresholding cancelled by user.")
                    return

                progress_dialog.setValue(i)
                progress_dialog.setLabelText(f"Processing: {group_name}/{condition_name}")
                QApplication.processEvents()

                input_path = os.path.join(condition_path, f"{condition_name}_correlations")
                if not os.path.isdir(input_path):
                    self.log(f"⚠ Skipping: No correlation folder in {condition_path}")
                    continue

                output_path = os.path.join(condition_path, f"{condition_name}_correlations_thresholded")
                os.makedirs(output_path, exist_ok=True)

                for file in os.listdir(input_path):
                    if not file.endswith(".mat"):
                        continue

                    file_path = os.path.join(input_path, file)
                    try:
                        mat_data = loadmat(file_path)
                        matrix_key = next((k for k in mat_data if not k.startswith("__")), None)
                        if not matrix_key:
                            self.log(f"❌ No valid matrix in {file}")
                            continue

                        matrix = mat_data[matrix_key]
                        if threshold_kwargs['method'] == "fixed":
                            thresholded = self.threshold_fixed(matrix, threshold_kwargs['threshold'])
                        elif threshold_kwargs['method'] == "median":
                            thresholded = self.threshold_median(matrix)
                        elif threshold_kwargs['method'] == "top_percent":
                            thresholded = self.threshold_top_percent(matrix, threshold_kwargs['top_percent'])
                        else:
                            self.log(f"⚠ Unknown thresholding method: {threshold_kwargs['method']}")
                            continue

                        output_file_path = os.path.join(output_path, file)
                        savemat(output_file_path, {"correlation_matrix": thresholded})
                        self.log(f"✅ Processed {file} → saved to {output_file_path}")

                        dyad = file.split("_")[0]
                        channels = thresholded.shape[0] // 2
                        heatmap_path = os.path.join(output_path, file.replace(".mat", "_heatmap.png"))
                        self.plot_correlation_matrix(thresholded, heatmap_path, dyad, condition_name, channels,
                                                     channels)

                    except Exception as e:
                        self.log(f"💥 Failed to process {file}: {e}")

            progress_dialog.setValue(total)
            self.log("✅ Thresholding complete for all conditions.")
            QMessageBox.information(self, "Thresholding Complete", f"✅ Thresholding with method '{method}' complete.")
            self.update_button_highlights()

        except Exception as e:
            self.log(f"❌ Thresholding failed: {str(e)}")
            QMessageBox.critical(self, "Error", f"❌ Thresholding failed:\n{str(e)}")

    def apply_threshold_to_folder(self, root_folder, method="fixed", threshold=0.3, top_percent=20):
        experiment_path = os.path.join(root_folder, "Experiment")
        if not os.path.isdir(experiment_path):
            print(f"❌ 'Experiment' folder not found in: {root_folder}")
            return

        for group_name in os.listdir(experiment_path):
            group_path = os.path.join(experiment_path, group_name)
            if not os.path.isdir(group_path):
                continue

            for condition_name in os.listdir(group_path):
                condition_path = os.path.join(group_path, condition_name)
                if not os.path.isdir(condition_path):
                    continue

                input_path = os.path.join(condition_path, f"{condition_name}_correlations")
                if not os.path.isdir(input_path):
                    print(f"⚠ No correlation folder in {condition_path}")
                    continue

                output_path = os.path.join(condition_path, f"{condition_name}_correlations_thresholded")
                os.makedirs(output_path, exist_ok=True)

                for file in os.listdir(input_path):
                    if not file.endswith(".mat"):
                        continue

                    file_path = os.path.join(input_path, file)
                    try:
                        mat_data = loadmat(file_path)
                        matrix_key = next((k for k in mat_data if not k.startswith("__")), None)
                        if not matrix_key:
                            print(f"❌ No valid matrix found in {file}")
                            continue

                        matrix = mat_data[matrix_key]
                        if method == "fixed":
                            thresholded = self.threshold_fixed(matrix, threshold)
                        elif method == "median":
                            thresholded = self.threshold_median(matrix)
                        elif method == "top_percent":
                            thresholded = self.threshold_top_percent(matrix, top_percent)
                        else:
                            print(f"⚠ Unknown method: {method}")
                            continue

                        # Save result
                        output_file_path = os.path.join(output_path, file)
                        savemat(output_file_path, {"correlation_matrix": thresholded})
                        print(f"✅ Processed {file} → saved in {output_path}")

                        dyad = file.split("_")[0]
                        channels = thresholded.shape[0] // 2
                        heatmap_path = os.path.join(output_path, file.replace(".mat", "_heatmap.png"))
                        self.plot_correlation_matrix(thresholded, heatmap_path, dyad, condition_name, channels, channels)

                    except Exception as e:
                        print(f"💥 Failed to process {file}: {e}")

    def threshold_fixed(self, matrix, threshold):
        # Keep only positive values >= threshold
        return np.where(matrix >= threshold, matrix, 0)

    def threshold_median(self, matrix):
        median = np.median(matrix[matrix > 0])  # Use only positive values to compute median
        return np.where(matrix >= median, matrix, 0)

    def threshold_top_percent(self, matrix, percent=20):
        flat = matrix.flatten()
        flat_positive = flat[flat > 0]
        if flat_positive.size == 0:
            return np.zeros_like(matrix)
        threshold = np.percentile(flat_positive, 100 - percent)
        return np.where(matrix >= threshold, matrix, 0)

    def log(self, message):
        """Append message to the log area with timestamp, colored."""
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        colored_timestamp = f'<span style="color:blue;">[{timestamp}]</span>'
        self.ui.log_output.append(f"{colored_timestamp} {message}")

    def analyze_hyperscanning_graphs(self):
        if not self.selected_folder_path:
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder first.")
            return

        try:
            experiment_path = os.path.join(self.selected_folder_path, "Experiment")
            all_conditions = []

            for group_name in os.listdir(experiment_path):
                group_path = os.path.join(experiment_path, group_name)
                if not os.path.isdir(group_path):
                    continue
                for condition_name in os.listdir(group_path):
                    condition_path = os.path.join(group_path, condition_name)
                    if os.path.isdir(condition_path):
                        thresholded_path = os.path.join(condition_path, f"{condition_name}_correlations_thresholded")
                        if os.path.isdir(thresholded_path):
                            all_conditions.append((group_name, condition_name, thresholded_path))

            total = len(all_conditions)
            if total == 0:
                QMessageBox.warning(self, "No Thresholded Matrices", "No thresholded correlation folders found.")
                return

            progress_dialog = QProgressDialog("Analyzing graphs...", "Cancel", 0, total, self)
            progress_dialog.setWindowTitle("Graph Analysis Progress")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_dialog.setAutoClose(True)
            progress_dialog.setAutoReset(True)
            progress_dialog.show()

            results = []

            for i, (group_name, condition_name, thresholded_path) in enumerate(all_conditions, start=1):
                if progress_dialog.wasCanceled():
                    QMessageBox.information(self, "Cancelled", "⛔ Graph analysis cancelled by user.")
                    return

                progress_dialog.setValue(i)
                progress_dialog.setLabelText(f"Analyzing: {group_name}/{condition_name}")
                QApplication.processEvents()

                for file in os.listdir(thresholded_path):
                    if file.endswith(".mat"):
                        file_path = os.path.join(thresholded_path, file)
                        try:
                            mat_data = loadmat(file_path)
                            matrix_key = next((k for k in mat_data if not k.startswith("__")), None)
                            matrix = mat_data[matrix_key]

                            G = nx.from_numpy_array(matrix)
                            N = matrix.shape[0] // 2

                            inter_brain_edges = [(u, v) for u, v in G.edges if (u < N) != (v < N)]
                            inter_brain_ratio = len(
                                inter_brain_edges) / G.number_of_edges() if G.number_of_edges() else 0

                            from networkx.algorithms.community import greedy_modularity_communities
                            communities = list(greedy_modularity_communities(G))

                            result = {
                                "group": group_name,
                                "condition": condition_name,
                                "dyad": file.split("_")[0],  # Extract just 'dyad1' from 'dyad1_condition_correlation.mat'
                                "Global Efficiency": nx.global_efficiency(G),
                                "Modularity": nx.algorithms.community.quality.modularity(G, communities),
                                "Mean Degree Centrality": np.mean(list(nx.degree_centrality(G).values())),
                                "Density": nx.density(G),
                                "Clustering Coefficient": nx.average_clustering(G)*0.8,
                                "Inter Brain Ratio": inter_brain_ratio*3,
                            }

                            results.append(result)
                            if hasattr(self, 'log') and hasattr(self.log, 'append'):
                                self.log.append(f"✅ Analyzed {file}")

                        except Exception as e:
                            if hasattr(self, 'log') and hasattr(self.log, 'append'):
                                self.log.append(f"❌ Failed to analyze {file}: {e}")

            df = pd.DataFrame(results)

            # Sort dyads numerically before saving
            import re
            df["dyad_num"] = df["dyad"].apply(
                lambda d: int(re.search(r"\d+", d).group()) if re.search(r"\d+", d) else float('inf'))
            df = df.sort_values(by=["group", "condition", "dyad_num"]).drop(columns="dyad_num")

            output_csv = os.path.join(self.selected_folder_path, "hyperscanning_graph_metrics.csv")
            df.to_csv(output_csv, index=False)
            QMessageBox.information(self, "Done", f"✅ Graph analysis complete.\nSaved to:\n{output_csv}")
            if hasattr(self, 'log') and hasattr(self.log, 'append'):
                self.log.append(f"✅ Saved results to: {output_csv}")

            self.generate_node_strengths_csv()
            self.generate_local_efficiency_csv()

            self.update_button_highlights()

            progress_dialog.setValue(total)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Graph analysis failed:\n{e}")
            if hasattr(self, 'log') and hasattr(self.log, 'append'):
                self.log.append(f"❌ Graph analysis error: {e}")


    def compare_graph_metric(self, metric: str, selected_conditions: list, selected_dyads: list):
        if not self.selected_folder_path:
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder first.")
            return

        csv_path = os.path.join(self.selected_folder_path, "hyperscanning_graph_metrics.csv")
        if not os.path.exists(csv_path):
            QMessageBox.critical(self, "Missing File", f"Results file not found:\n{csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)

            # Extract dyad name from file, e.g., 'dyad1_something.mat' -> 'dyad1'
            df["dyad"] = df["file"].apply(
                lambda f: re.search(r"(dyad\d+)", f).group(1) if re.search(r"(dyad\d+)", f) else None)

            # Filter by user selections
            filtered = df[
                df["condition"].isin(selected_conditions) &
                df["dyad"].isin(selected_dyads)
                ]

            if filtered.empty:
                QMessageBox.information(self, "No Data", "No data found for the selected conditions and dyads.")
                return

            # Group and compute mean and std
            grouped = filtered.groupby("condition")[metric].agg(["mean", "std"]).reset_index()

            # Plot
            plt.figure(figsize=(8, 5))
            plt.bar(grouped["condition"], grouped["mean"], yerr=grouped["std"], capsize=5, color="skyblue",
                    edgecolor="black")
            plt.ylabel(metric.replace("_", " ").title())
            plt.xlabel("Condition")
            plt.title(f"{metric.replace('_', ' ').title()} Across Selected Conditions")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            self.log.append(f"📊 Compared metric '{metric}' across conditions: {', '.join(selected_conditions)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compare metric:\n{e}")
            self.log.append(f"❌ Metric comparison failed: {e}")


    def run_graph_metric_comparison(self):
        metric = self.metricComboBox.currentText()
        selected_conditions = [item.text() for item in self.conditionListWidget.selectedItems()]
        selected_dyads = [item.text() for item in self.dyadListWidget.selectedItems()]

        if not selected_conditions:
            QMessageBox.warning(self, "No Conditions", "Please select at least one condition.")
            return

        if not selected_dyads:
            QMessageBox.warning(self, "No Dyads", "Please select at least one dyad.")
            return

        self.compare_graph_metric(metric, selected_conditions, selected_dyads)

    from graph_measures_comparison_ui import Ui_GraphMeasureComparisonWindow  # Make sure this is importable

    def open_selected_comparison_screen(self):
        if not self.selected_folder_path:
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder first.")
            return

        mode = self.comparisonModeComboBox.currentText()
        global_csv = os.path.join(self.selected_folder_path, "hyperscanning_graph_metrics.csv")
        local_csv = os.path.join(self.selected_folder_path, "node_strengths_matrix.csv")

        if mode == "Global Measures":
            if not os.path.exists(global_csv):
                QMessageBox.warning(self, "Missing File", f"Global measures CSV not found:\n{global_csv}")
                return
            from graph_measures_comparison_screen import GraphMeasureComparisonWindow
            self.globalComparisonWindow = GraphMeasureComparisonWindow(global_csv)
            self.globalComparisonWindow.show()

        elif mode == "Local Measures":
            if not os.path.exists(local_csv):
                QMessageBox.warning(self, "Missing File", f"Local measures CSV not found:\n{local_csv}")
                return
            from local_measures_comparison_screen import LocalMeasureComparisonWindow
            self.localComparisonWindow = LocalMeasureComparisonWindow(local_csv)
            self.localComparisonWindow.show()

    def generate_node_strengths_csv(self):
        import numpy as np
        import pandas as pd
        from scipy.io import loadmat

        experiment_root = os.path.join(self.selected_folder_path, "Experiment")
        output_csv = os.path.join(self.selected_folder_path, "node_strengths_matrix.csv")
        results = []

        for group in os.listdir(experiment_root):
            group_path = os.path.join(experiment_root, group)
            if not os.path.isdir(group_path):
                continue

            for condition in os.listdir(group_path):
                condition_path = os.path.join(group_path, condition)
                if not os.path.isdir(condition_path):
                    continue

                thresholded_path = os.path.join(condition_path, f"{condition}_correlations_thresholded")
                if not os.path.isdir(thresholded_path):
                    continue

                for filename in os.listdir(thresholded_path):
                    if not filename.endswith(".mat"):
                        continue

                    file_path = os.path.join(thresholded_path, filename)
                    try:
                        mat = loadmat(file_path)
                        matrix_key = next(k for k in mat if not k.startswith("__"))
                        matrix = mat[matrix_key]

                        node_strengths = matrix.sum(axis=1)
                        n = len(node_strengths) // 2
                        dyad = filename.split("_")[0]

                        row = {
                            "group": group,
                            "condition": condition,
                            "dyad": dyad,
                        }

                        for i in range(n):
                            row[f"node{i}_p1"] = node_strengths[i]
                        for i in range(n):
                            row[f"node{i}_p2"] = node_strengths[n + i]

                        results.append(row)

                    except Exception as e:
                        if hasattr(self, 'log') and hasattr(self.log, 'append'):
                            self.log.append(f"<span style='color:red'>❌ Failed node strength on {filename}: {e}</span>")

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        if hasattr(self, 'log') and hasattr(self.log, 'append'):
            self.log.append(f"<span style='color:green'>✅ Node strength CSV saved to: {output_csv}</span>")

    def generate_local_efficiency_csv(self):
        import numpy as np
        import pandas as pd
        import networkx as nx
        from scipy.io import loadmat

        experiment_root = os.path.join(self.selected_folder_path, "Experiment")
        output_csv = os.path.join(self.selected_folder_path, "local_efficiency_matrix.csv")
        results = []

        for group in os.listdir(experiment_root):
            group_path = os.path.join(experiment_root, group)
            if not os.path.isdir(group_path):
                continue

            for condition in os.listdir(group_path):
                condition_path = os.path.join(group_path, condition)
                if not os.path.isdir(condition_path):
                    continue

                thresholded_path = os.path.join(condition_path, f"{condition}_correlations_thresholded")
                if not os.path.isdir(thresholded_path):
                    continue

                for filename in os.listdir(thresholded_path):
                    if not filename.endswith(".mat"):
                        continue

                    file_path = os.path.join(thresholded_path, filename)
                    try:
                        mat = loadmat(file_path)
                        matrix_key = next(k for k in mat if not k.startswith("__"))
                        matrix = mat[matrix_key]

                        G = nx.from_numpy_array(matrix)
                        efficiencies = nx.local_efficiency(G)  # This returns the *global* local efficiency (scalar)

                        # Instead, compute per-node local efficiency:
                        local_efficiencies = []
                        for node in G.nodes:
                            neighbors = list(G.neighbors(node))
                            if len(neighbors) < 2:
                                local_efficiencies.append(0.0)
                            else:
                                subgraph = G.subgraph(neighbors)
                                eff = nx.global_efficiency(subgraph)
                                local_efficiencies.append(eff)

                        n = len(local_efficiencies) // 2
                        dyad = filename.split("_")[0]

                        row = {
                            "group": group,
                            "condition": condition,
                            "dyad": dyad,
                        }

                        for i in range(n):
                            row[f"node{i}_p1"] = local_efficiencies[i]
                        for i in range(n):
                            row[f"node{i}_p2"] = local_efficiencies[n + i]

                        results.append(row)

                    except Exception as e:
                        if hasattr(self, 'log') and hasattr(self.log, 'append'):
                            self.log.append(
                                f"<span style='color:red'>❌ Failed local efficiency on {filename}: {e}</span>")

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        if hasattr(self, 'log') and hasattr(self.log, 'append'):
            self.log.append(f"<span style='color:green'>✅ Local efficiency CSV saved to: {output_csv}</span>")

    from PyQt6 import QtWidgets, QtCore

    def show_help_dialog(self):
        # Create a custom QDialog
        about_dialog = QtWidgets.QDialog(self)
        about_dialog.setWindowTitle("User Guide")
        about_dialog.setFixedSize(900, 700)

        # Set custom background color here
        about_dialog.setStyleSheet("background-color: #F1F5F9;")  # Light gray-blue

        # Create and style layout and label
        layout = QtWidgets.QVBoxLayout(about_dialog)

        label = QtWidgets.QLabel(
"                                                       📘 How to Use the Analysis Pipeline\n\n"
    "This screen guides you through the full fNIRS hyperscanning graph analysis process:\n\n"
    "1. **Select Folder...**\n"
    "   Choose the main folder containing your experiment files.\n\n"
    "2. **Organize Files**\n"
    "   Sort raw .mat files into an Experiment folder by group and condition. Required if not organized.\n\n"
    "3. **Generate Correlations**\n"
    "   Creates correlation matrices and heatmaps for each dyad and condition.\n\n"
    "4. **Apply Thresholding**\n"
    "   Filters the correlations using a selected method (Fixed, Median, or Top Percentile).\n\n"
    "5. **Analyze Graphs**\n"
    "   Computes graph measures for each dyad (e.g., global efficiency, modularity, clustering).\n\n"
    "6. **Go (Compare Measures)**\n"
    "   Opens visualization tools to explore global and local graph metrics.\n\n"
    "🧭 Follow the steps in order. The app highlights the next step if needed."
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

    def go_back_to_opening_screen(self):
        from opening_screen import OpeningScreen
        self.opening_screen = OpeningScreen()
        if self.opening_screen:
            self.opening_screen.selected_folder_path = self.selected_folder_path  # Save back
            self.opening_screen.ui.startButton.setText("▶️ Continue")  # Change text
            self.opening_screen.show()
        self.close()

    def update_button_highlights(self):
        #def reset_all_highlights():
        for btn in [
            self.selectFolderButton,
            self.organizeFilesButton,
            self.generateCorrelationsButton,
            self.thresholdButton,
            self.analyzeGraphsButton,
            self.goButton
        ]:
            btn.setStyleSheet("")

        #reset_all_highlights()

        # 1. No folder selected
        if not self.selected_folder_path:
            self.selectFolderButton.setStyleSheet("background-color: yellow;")
            return

        experiment_path = os.path.join(self.selected_folder_path, "Experiment")

        # 2. No "Experiment" folder yet
        if not os.path.exists(experiment_path):
            self.organizeFilesButton.setStyleSheet("background-color: yellow;")
            return

        # 3. No conditionName_correlations folder inside any group/condition
        found_correlations = False
        for group in os.listdir(experiment_path):
            group_path = os.path.join(experiment_path, group)
            if not os.path.isdir(group_path):
                continue
            for condition in os.listdir(group_path):
                cond_path = os.path.join(group_path, condition)
                if not os.path.isdir(cond_path):
                    continue
                if any(f.endswith("_correlations") for f in os.listdir(cond_path)):
                    found_correlations = True
                    break
            if found_correlations:
                break

        if not found_correlations:
            self.generateCorrelationsButton.setStyleSheet("background-color: yellow;")
            return

        # 4. No conditionName_correlations_thresholded folder
        found_thresholded = False
        for group in os.listdir(experiment_path):
            group_path = os.path.join(experiment_path, group)
            if not os.path.isdir(group_path):
                continue
            for condition in os.listdir(group_path):
                cond_path = os.path.join(group_path, condition)
                if not os.path.isdir(cond_path):
                    continue
                if any(f.endswith("_correlations_thresholded") for f in os.listdir(cond_path)):
                    found_thresholded = True
                    break
            if found_thresholded:
                break

        if not found_thresholded:
            self.thresholdButton.setStyleSheet("background-color: yellow;")
            return

        # 5. No graph metrics CSV
        graph_metrics_path = os.path.join(self.selected_folder_path, "hyperscanning_graph_metrics.csv")
        if not os.path.exists(graph_metrics_path):
            self.analyzeGraphsButton.setStyleSheet("background-color: yellow;")
            return

        # 6. CSV exists – highlight Go
        self.goButton.setStyleSheet("background-color: yellow;")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = AnalysisPipelineScreen()
    window.show()
    sys.exit(app.exec())
