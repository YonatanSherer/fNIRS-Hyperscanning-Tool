from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import platform
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

class GraphMeasureComparisonWindow(QtWidgets.QMainWindow):
    def __init__(self, csv_path, parent=None):
        super().__init__(parent)

        from graph_measures_comparison_ui import Ui_GraphMeasureComparisonWindow
        self.ui = Ui_GraphMeasureComparisonWindow()
        self.ui.setupUi(self)

        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)

        # Ensure 'dyad' column exists, or extract it from 'file'
        if 'dyad' not in self.df.columns:
            if 'file' in self.df.columns:
                import re
                self.df['dyad'] = self.df['file'].apply(
                    lambda f: re.match(r'(dyad\d+)', f).group(1) if pd.notnull(f) and re.match(r'(dyad\d+)', f) else ''
                )
            else:
                QMessageBox.critical(self, "Missing Columns", "The CSV is missing both 'dyad' and 'file' columns.")
                self.close()
                return

        # Validate required columns
        required_columns = {'group', 'condition', 'dyad'}
        missing = required_columns - set(self.df.columns)
        if missing:
            QMessageBox.critical(self, "Missing Columns", f"The CSV is missing: {', '.join(missing)}")
            self.close()
            return

        #Initially Disable the Button
        self.openPlotImageButton = self.findChild(QtWidgets.QPushButton, "openPlotImageButton")
        self.openPlotImageButton.setEnabled(False)
        self.openPlotImageButton.clicked.connect(self.open_plot_as_image)

        # Widgets
        self.groupListWidget = self.findChild(QtWidgets.QListWidget, "groupListWidget")
        self.dyadListWidget = self.findChild(QtWidgets.QListWidget, "dyadListWidget")
        self.conditionListWidget = self.findChild(QtWidgets.QListWidget, "conditionListWidget")
        self.graphMeasureComboBox = self.findChild(QtWidgets.QComboBox, "graphMeasureComboBox")
        self.compareButton = self.findChild(QtWidgets.QPushButton, "compareButton")
        self.plotLabel = self.findChild(QtWidgets.QLabel, "plotLabel")
        self.plotWidget = self.findChild(QtWidgets.QWidget, "plotWidget")
        self.backButton = self.findChild(QtWidgets.QPushButton, "backButton")

        self.chartTypeComboBox = self.findChild(QtWidgets.QComboBox, "chartTypeComboBox")
        self.chartTypeComboBox.addItems(["Line Chart", "Bar Chart", "Radar Chart"])

        self.chartTypeComboBox.currentTextChanged.connect(self.update_graph_measure_selection_state)

        self.compareButton.clicked.connect(self.compare_graph_measure)
        self.backButton.clicked.connect(self.go_back)

        self.selectAllDyadsButton = self.findChild(QtWidgets.QPushButton, "selectAllDyadsButton")
        self.clearDyadsButton = self.findChild(QtWidgets.QPushButton, "clearDyadsButton")

        self.selectAllDyadsButton.clicked.connect(self.select_all_dyads)
        self.clearDyadsButton.clicked.connect(self.clear_dyads_selection)

        self.openCsvButton = self.findChild(QtWidgets.QPushButton, "openCsvButton")
        self.openCsvButton.clicked.connect(self.open_csv_file)

        self.openNodeStrengthsCsvButton = self.findChild(QtWidgets.QPushButton, "openNodeStrengthsCsvButton")
        self.openNodeStrengthsCsvButton.clicked.connect(self.open_node_strengths_csv)
        if self.openNodeStrengthsCsvButton:
            self.openNodeStrengthsCsvButton.hide()

        self.helpButton = self.findChild(QtWidgets.QPushButton, "helpButton")
        self.helpButton.clicked.connect(self.show_help_dialog)

        self.exportButton = self.findChild(QtWidgets.QPushButton, "exportButton")
        self.exportButton.setEnabled(False)
        self.exportButton.clicked.connect(self.export_results)

        self.populate_fields()


    def populate_fields(self):
        import re
        dyads = sorted(
            self.df['dyad'].dropna().unique(),
            key=lambda d: int(re.search(r'\d+', d).group()) if re.search(r'\d+', d) else float('inf')
        )

        conditions = sorted(self.df['condition'].dropna().unique())
        groups = sorted(self.df['group'].dropna().unique())
        excluded = {'group', 'condition', 'dyad', 'file'}
        metrics = sorted([col for col in self.df.columns if col not in excluded])

        self.dyadListWidget.clear()
        self.conditionListWidget.clear()
        self.groupListWidget.clear()
        self.graphMeasureComboBox.clear()

        # Build dyad → group mapping
        self.dyad_to_group = dict(self.df[['dyad', 'group']].dropna().values)

        # Populate dyads — initially disabled
        for dyad in dyads:
            item = QtWidgets.QListWidgetItem(str(dyad))
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)  # Initially disabled
            self.dyadListWidget.addItem(item)

        # Populate conditions
        for cond in conditions:
            item = QtWidgets.QListWidgetItem(str(cond))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.conditionListWidget.addItem(item)

        # Populate groups
        for group in groups:
            item = QtWidgets.QListWidgetItem(str(group))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.groupListWidget.addItem(item)

        # Connect group selection change to dyad filter
        self.groupListWidget.itemChanged.connect(self.update_dyad_checkboxes_by_group_selection)

        # Populate graph metrics
        self.graphMeasureComboBox.addItems(metrics)

    def compare_graph_measure(self):
        selected_dyads = [self.dyadListWidget.item(i).text() for i in range(self.dyadListWidget.count())
                          if self.dyadListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        selected_conditions = [self.conditionListWidget.item(i).text() for i in range(self.conditionListWidget.count())
                               if self.conditionListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        selected_groups = [self.groupListWidget.item(i).text() for i in range(self.groupListWidget.count())
                           if self.groupListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        selected_metric = self.graphMeasureComboBox.currentText()

        if not selected_dyads or not selected_conditions or not selected_groups or not selected_metric:
            QMessageBox.warning(
                self,
                "Missing selection",
                "Please select at least one group, one dyad, one condition, and a metric."
            )
            return

        filtered = self.df[
            self.df['dyad'].astype(str).isin(selected_dyads) &
            self.df['condition'].astype(str).isin(selected_conditions) &
            self.df['group'].astype(str).isin(selected_groups)
        ]

        if filtered.empty:
            QMessageBox.information(self, "No data", "No data matches the selection.")
            return

        self.plot_comparison(filtered, selected_metric)

        self.exportButton.setEnabled(True)

    def plot_comparison(self, data, metric):
        """Plot the selected graph metric across selected dyads and conditions.
           Supports dynamic filtering based on user selections in the UI."""
        chart_type = self.chartTypeComboBox.currentText() if hasattr(self, 'chartTypeComboBox') else "Line Chart"

        plt.clf()
        fig = Figure(figsize=(8, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        self.current_plot_figure = fig  #store for later use

        if chart_type == "Line Chart":
            for dyad in data['dyad'].unique():
                sub_data = data[data['dyad'] == dyad]
                ax.plot(sub_data['condition'], sub_data[metric], marker='o', label=f'Dyad {dyad}')


        elif chart_type == "Bar Chart":
            # Sort dyads numerically before plotting
            data["dyad_num"] = data["dyad"].str.extract(r"(\d+)").astype(int)
            data = data.sort_values("dyad_num")

            # Group by group+dyad, then unstack conditions into columns
            data['dyad_label'] = data['group'].astype(str) + "_" + data['dyad'].astype(str)
            grouped = data.groupby(['dyad_label', 'condition'])[metric].mean().unstack()

            # Calculate mean & std
            condition_means = grouped.mean(axis=0)
            condition_stds = grouped.std(axis=0)

            # Add average row
            grouped.loc['Average'] = condition_means

            # Plot bar chart
            grouped.plot(kind='bar', ax=ax)

            # Divider lines between groups
            group_names = [label.split('_')[0] for label in grouped.index]
            previous_group = group_names[0]

            for i in range(1, len(group_names)):
                current_group = group_names[i]
                if current_group != previous_group:
                    ax.axvline(i - 0.5, color='black', linestyle='--', linewidth=1)
                previous_group = current_group

            # Error bars on average row
            average_idx = list(grouped.index).index('Average')
            x_ticks = ax.get_xticks()
            bar_width = 0.8 / len(condition_means)
            offsets = [i * bar_width - 0.4 + bar_width / 2 for i in range(len(condition_means))]

            for i, (cond, std) in enumerate(condition_stds.items()):
                if pd.notnull(std):
                    ax.errorbar(
                        x=x_ticks[average_idx] + offsets[i],
                        y=condition_means[cond],
                        yerr=std,
                        fmt='none',
                        ecolor='black',
                        capsize=5,
                        elinewidth=1.5

                    )

            ax.set_title(f'{metric} (Bar Chart with Group Dividers & Std Dev)')

            if "dyad_num" in data.columns:
                data.drop(columns="dyad_num", inplace=True)


        elif chart_type == "Radar Chart":
            if len(data['dyad'].unique()) != 1 or len(data['condition'].unique()) != 1:
                QMessageBox.warning(
                    self,
                    "Radar Chart Limit",
                    "Please select exactly one dyad and one condition for radar chart."
                )
                return

            dyad = data['dyad'].iloc[0]
            condition = data['condition'].iloc[0]
            exclude = {'group', 'condition', 'dyad', 'file'}
            metrics = [col for col in data.columns if col not in exclude]
            values = data.iloc[0][metrics].astype(float).values.tolist()

            # Normalize values to [0, 1]
            min_val = min(values)
            max_val = max(values)

            if max_val - min_val > 1e-8:
                values = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                values = [0.0 for _ in values]  # fallback in degenerate case

            # Repeat first value/metric to close the radar chart
            metrics += [metrics[0]]
            values += [values[0]]
            angles = [n / float(len(metrics)) * 2 * np.pi for n in range(len(metrics))]

            # Radar chart
            ax = fig.add_subplot(111, polar=True)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'{dyad} - {condition}')
            ax.fill(angles, values, alpha=0.3)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics[:-1], fontsize=9)
            ax.set_yticklabels([])
            ax.set_title(f"Normalized Radar Chart: {dyad} - {condition}")

        if chart_type == "Radar Chart":
            ax.set_title(f"Graph Measures Radar Chart for {dyad} - {condition}")
        else:
            ax.set_title(f'{metric} ({chart_type})')

        if chart_type != "Radar Chart":
            ax.set_xlabel("Condition")
            ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

        layout = self.plotWidget.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(self.plotWidget)
        else:
            self.clear_layout(layout)

        layout.addWidget(canvas)

        self.openPlotImageButton.setEnabled(True)  #Enable the button when plot is created

        if hasattr(self, "plotLabel"):
            self.plotLabel.hide()

    def update_graph_measure_selection_state(self, chart_type):
        if chart_type == "Radar Chart":
            self.graphMeasureComboBox.setEnabled(False)
        else:
            self.graphMeasureComboBox.setEnabled(True)

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

    def go_back(self):
        self.close()

    def select_all_dyads(self):
        for i in range(self.dyadListWidget.count()):
            item = self.dyadListWidget.item(i)
            if item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(QtCore.Qt.CheckState.Checked)
            else:
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def clear_dyads_selection(self):
        for i in range(self.dyadListWidget.count()):
            item = self.dyadListWidget.item(i)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def open_plot_as_image(self):
        if not hasattr(self, "current_plot_figure"):
            QMessageBox.warning(self, "No Plot", "No plot is currently available.")
            return

        import tempfile
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl

        # Save the figure to a temporary PNG file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            self.current_plot_figure.savefig(tmp_file.name, bbox_inches='tight')
            tmp_file_path = tmp_file.name

        # Open the file with the default image viewer
        QDesktopServices.openUrl(QUrl.fromLocalFile(tmp_file_path))

    def open_csv_file(self):
        if os.path.exists(self.csv_path):
            if platform.system() == "Windows":
                os.startfile(self.csv_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(["open", self.csv_path])
            else:  # Linux and others
                subprocess.call(["xdg-open", self.csv_path])
        else:
            QMessageBox.warning(self, "File Not Found", "CSV file not found.")

    def open_node_strengths_csv(self):
        node_csv_path = os.path.join(os.path.dirname(self.csv_path), "node_strengths_matrix.csv")
        if os.path.exists(node_csv_path):
            if platform.system() == "Windows":
                os.startfile(node_csv_path)
            elif platform.system() == "Darwin":
                subprocess.call(["open", node_csv_path])
            else:
                subprocess.call(["xdg-open", node_csv_path])
        else:
            QMessageBox.warning(self, "File Not Found", "Node strengths CSV file not found.")

    def update_dyad_checkboxes_by_group_selection(self):
        selected_groups = {
            self.groupListWidget.item(i).text()
            for i in range(self.groupListWidget.count())
            if self.groupListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked
        }

        for i in range(self.dyadListWidget.count()):
            dyad_item = self.dyadListWidget.item(i)
            dyad_name = dyad_item.text()
            dyad_group = self.dyad_to_group.get(dyad_name)

            if dyad_group in selected_groups:
                dyad_item.setFlags(
                    dyad_item.flags() | QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            else:
                dyad_item.setFlags(dyad_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
                dyad_item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def show_help_dialog(self):
        # Create a custom QDialog
        about_dialog = QtWidgets.QDialog(self)
        about_dialog.setWindowTitle("Global Measures Comparison - Help")
        about_dialog.setFixedSize(800, 600)

        # Set custom background color here
        about_dialog.setStyleSheet("background-color: #F1F5F9;")  # Light gray-blue

        # Create and style layout and label
        layout = QtWidgets.QVBoxLayout(about_dialog)

        label = QtWidgets.QLabel(
            "                                     🧠 Global Measures Comparison – Help Guide\n\n"
    "This screen lets you explore and compare global graph metrics such as Global Efficiency, "
    "Modularity, Degree Centrality, and more across dyads and conditions.\n\n"
    "✅ How to Use:\n"
    "1. Select Groups, Conditions, and Dyads to include in the comparison.\n"
    "2. Choose a global graph measure from the dropdown (or select 'Radar Chart' to view all).\n"
    "3. Select a chart type (Bar Chart, Line Chart, or Radar Chart).\n"
    "4. Click \"Compare\" to generate the plot.\n\n"
    "📊 Recommended Analysis Options:\n"
    "⭐ Compare a specific measure (e.g., Modularity) across conditions or dyads.\n"
    "⭐ Use Radar Chart to visualize the overall profile of each condition.\n"
    "⭐ Select multiple dyads from the same group to examine inter-dyad variability.\n\n"
    "📌 Tip:\n"
    "The \"Average\" bar summarizes selected dyads per condition.\n\n"
    "📤 Use the \"Export\" button to save the current plot, data, and summary as a report."
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

    def normalize_measures(df, columns):
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df

    def export_results(self):
        import os
        import pandas as pd
        from PyQt6.QtWidgets import QMessageBox
        from datetime import datetime

        if not hasattr(self, 'current_plot_figure') or self.df.empty:
            QMessageBox.warning(self, "Export Error", "Please compare data before exporting.")
            return

        # Get export folder path
        base_folder = os.path.dirname(self.csv_path)
        export_dir = os.path.join(base_folder, "Results", "Global Measures")
        os.makedirs(export_dir, exist_ok=True)

        # Current selections
        selected_groups = [self.groupListWidget.item(i).text() for i in range(self.groupListWidget.count())
                           if self.groupListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        selected_conditions = [self.conditionListWidget.item(i).text() for i in range(self.conditionListWidget.count())
                               if self.conditionListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        selected_dyads = [self.dyadListWidget.item(i).text() for i in range(self.dyadListWidget.count())
                          if self.dyadListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        chart_type = self.chartTypeComboBox.currentText()

        # For radar chart we treat metric = "All"
        selected_metric = "All" if chart_type == "Radar Chart" else self.graphMeasureComboBox.currentText()

        # Validate column exists (non-radar case)
        if chart_type != "Radar Chart" and selected_metric not in self.df.columns:
            QMessageBox.critical(self, "Export Error",
                                 f"The selected metric '{selected_metric}' is not found in the data.")
            return

        # Filter original DataFrame
        filtered = self.df[
            self.df['group'].astype(str).isin(selected_groups) &
            self.df['condition'].astype(str).isin(selected_conditions) &
            self.df['dyad'].astype(str).isin(selected_dyads)
            ]

        if filtered.empty:
            QMessageBox.warning(self, "No Data", "Nothing to export. Please review your selections.")
            return

        try:
            # Create a unique subfolder using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dedicated_folder = os.path.join(export_dir, f"Export_{timestamp}")
            os.makedirs(dedicated_folder, exist_ok=True)

            # Base filename
            filename_base = f"{selected_metric.replace(' ', '_')}_{chart_type.replace(' ', '_')}".lower()

            # CSV
            csv_path = os.path.join(dedicated_folder, f"{filename_base}.csv")
            cols_to_save = ['group', 'condition', 'dyad']
            if chart_type != "Radar Chart":
                cols_to_save.append(selected_metric)
            else:
                # save all global measures for radar
                exclude = {'group', 'condition', 'dyad', 'file'}
                cols_to_save = [c for c in filtered.columns if c not in exclude] + ['group', 'condition', 'dyad']
            filtered[cols_to_save].to_csv(csv_path, index=False)

            # PNG
            img_path = os.path.join(dedicated_folder, f"{filename_base}.png")
            self.current_plot_figure.savefig(img_path, bbox_inches='tight')

            # TXT summary
            summary_path = os.path.join(dedicated_folder, "summary.txt")

            # Auto interpretation
            interpretation = self._build_interpretation_global(filtered, chart_type, selected_metric)

            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("Export Summary\n")
                f.write("==============\n\n")
                f.write(f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Selected Metric: {selected_metric}\n")
                f.write(f"Chart Type: {chart_type}\n\n")
                f.write(f"Selected Groups: {', '.join(selected_groups)}\n")
                f.write(f"Selected Conditions: {', '.join(selected_conditions)}\n")
                f.write(f"Selected Dyads: {', '.join(selected_dyads)}\n\n")
                f.write("Files:\n")
                f.write(f"- {os.path.basename(csv_path)}\n")
                f.write(f"- {os.path.basename(img_path)}\n\n")
                f.write("Interpretation:\n")
                f.write(interpretation + "\n")

            # PDF
            pdf_path = os.path.join(dedicated_folder, f"{filename_base}.pdf")
            selections = {
                "Metric": selected_metric,
                "Chart Type": chart_type,
                "Groups": ", ".join(selected_groups),
                "Conditions": ", ".join(selected_conditions),
                "Dyads": ", ".join(selected_dyads),
            }
            self._export_pdf(pdf_path, "Global Measures Export", selections, img_path, interpretation)

            QMessageBox.information(self, "Export Complete",
                                    f"CSV, image, text, and PDF exported to:\n{dedicated_folder}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred during export:\n{e}")



    def _build_interpretation_global(self, filtered: pd.DataFrame, chart_type: str, selected_metric: str) -> str:
        """Return a short human-readable interpretation for the global measures plot."""
        lines = []
        if chart_type == "Radar Chart":
            # We plotted all measures, so provide a general statement
            lines.append("Radar chart compares all global measures for the selected dyad & condition.")
            # Heuristic: tell which metric is max among the measures for that dyad/condition
            exclude = {'group', 'condition', 'dyad', 'file', 'dyad_label'}
            metrics_cols = [c for c in filtered.columns if c not in exclude]
            if len(filtered) > 0 and metrics_cols:
                row = filtered.iloc[0]
                best = max(metrics_cols, key=lambda m: row[m])
                lines.append(f"The strongest (highest) measure is '{best}' (value={row[best]:.3f}).")
            return "\n".join(lines)

        # Non-radar: single metric
        if selected_metric not in filtered.columns:
            return "Selected metric not found in data. Cannot interpret."

        # Basic stats by condition
        by_cond = filtered.groupby("condition")[selected_metric]
        means = by_cond.mean().sort_values(ascending=False)
        top_cond, top_val = means.index[0], means.iloc[0]
        lines.append(f"'{top_cond}' shows the highest average {selected_metric} ({top_val:.3f}).")

        # Variability
        stds = by_cond.std().sort_values(ascending=False)
        if not stds.empty and not np.isnan(stds.iloc[0]):
            lines.append(f"'{stds.index[0]}' shows the greatest variability (STD={stds.iloc[0]:.3f}).")

        # Simple trend (first vs last condition in sorted order)
        ordered = by_cond.mean().reindex(sorted(by_cond.mean().index))
        if len(ordered) >= 2:
            trend = "increasing" if ordered.iloc[-1] > ordered.iloc[0] else "decreasing"
            lines.append(f"A generally {trend} trend is observed from {ordered.index[0]} to {ordered.index[-1]}.")

        return "\n".join(lines)

    def _export_pdf(self, pdf_path: str, title: str, selections: dict, plot_image_path: str, interpretation: str):
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []

        story.append(Paragraph(title, styles["Title"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("<b>Selections</b>", styles["Heading3"]))
        for k, v in selections.items():
            story.append(Paragraph(f"<b>{k}:</b> {v}", styles["BodyText"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("<b>Plot</b>", styles["Heading3"]))
        story.append(Image(plot_image_path, width=480, height=300))  # adjust as you like
        story.append(Spacer(1, 12))

        story.append(Paragraph("<b>Auto-generated Interpretation</b>", styles["Heading3"]))
        story.append(Paragraph(interpretation.replace("\n", "<br/>"), styles["BodyText"]))

        doc.build(story)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = GraphMeasureComparisonWindow("hyperscanning_graph_metrics.csv")
    window.show()
    sys.exit(app.exec())
