from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import pandas as pd
import os
import subprocess
import platform
import numpy as np
from fpdf import FPDF
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

class LocalMeasureComparisonWindow(QtWidgets.QMainWindow):
    def __init__(self, csv_path, parent=None):
        super().__init__(parent)
        self.parent_window = parent  # Store the parent window

        from local_measures_comparison_ui import Ui_LocalMeasureComparisonWindow
        self.ui = Ui_LocalMeasureComparisonWindow()
        self.ui.setupUi(self)

        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)

        required_columns = {'group', 'condition', 'dyad'}
        missing = required_columns - set(self.df.columns)
        if missing:
            QMessageBox.critical(self, "Missing Columns", f"The CSV is missing: {', '.join(missing)}")
            self.close()
            return

        # Widgets
        self.groupListWidget = self.findChild(QtWidgets.QListWidget, "groupListWidget")
        self.dyadListWidget = self.findChild(QtWidgets.QListWidget, "dyadListWidget")
        self.conditionListWidget = self.findChild(QtWidgets.QListWidget, "conditionListWidget")
        self.nodeListWidget = self.findChild(QtWidgets.QListWidget, "nodeMeasureListWidget")
        self.compareButton = self.findChild(QtWidgets.QPushButton, "compareButton")
        self.plotWidget = self.findChild(QtWidgets.QWidget, "plotWidget")
        self.plotLabel = self.findChild(QtWidgets.QLabel, "plotLabel")
        self.openCsvButton = self.findChild(QtWidgets.QPushButton, "openCsvButton")
        self.selectAllNodesButton = self.findChild(QtWidgets.QPushButton, "selectAllNodesButton")
        self.clearNodesButton = self.findChild(QtWidgets.QPushButton, "clearNodesButton")
        self.chartTypeComboBox = self.findChild(QtWidgets.QComboBox, "chartTypeComboBox")
        self.metricComboBox = self.findChild(QtWidgets.QComboBox, "metricComboBox")

        # Connect signals
        self.compareButton.clicked.connect(self.compare_node_strength)
        self.openCsvButton.clicked.connect(self.open_csv_file)
        self.selectAllNodesButton.clicked.connect(self.select_all_nodes)
        self.clearNodesButton.clicked.connect(self.clear_nodes_selection)

        self.selectAllDyadsButton = self.findChild(QtWidgets.QPushButton, "selectAllDyadsButton")
        self.clearDyadsButton = self.findChild(QtWidgets.QPushButton, "clearDyadsButton")

        self.selectAllDyadsButton.clicked.connect(self.select_all_dyads)
        self.clearDyadsButton.clicked.connect(self.clear_dyads_selection)

        self.openPlotImageButton = self.findChild(QtWidgets.QPushButton, "openPlotImageButton")
        self.openPlotImageButton.setEnabled(False)  # Initially disabled
        self.openPlotImageButton.clicked.connect(self.open_plot_as_image)

        if self.metricComboBox:
            self.metricComboBox.clear()
            self.metricComboBox.addItems(["Node Strength", "Local Efficiency"])
            self.metricComboBox.currentIndexChanged.connect(self.metric_changed)

        if self.chartTypeComboBox:
            self.chartTypeComboBox.clear()
            self.chartTypeComboBox.addItems(["Bar Chart", "Line Chart"])

        self.populate_fields()

        self.group_colors = {}  # Group → QColor
        self.available_colors = [
            QtGui.QColor("#ffcccc"),  # Light red
            QtGui.QColor("#ccffcc"),  # Light green
            QtGui.QColor("#ccccff"),  # Light blue
            QtGui.QColor("#ffffcc"),  # Light yellow
            QtGui.QColor("#e0ccff"),  # Light purple
            QtGui.QColor("#ffd9cc"),  # Light orange
        ]

        self.backButton = self.findChild(QtWidgets.QPushButton, "backButton")
        if self.backButton:
            self.backButton.clicked.connect(self.go_back)

        self.helpButton = self.findChild(QtWidgets.QPushButton, "helpButton")
        if self.helpButton:
            self.helpButton.clicked.connect(self.show_help_dialog)

        self.exportButton = self.findChild(QtWidgets.QPushButton, "exportButton")
        self.exportButton.setEnabled(False)
        self.exportButton.clicked.connect(self.export_results)

    def populate_fields(self):
        self.groupListWidget.clear()
        self.dyadListWidget.clear()
        self.conditionListWidget.clear()
        self.nodeListWidget.clear()

        # Add this to ensure chart types are populated
        if self.chartTypeComboBox:
            self.chartTypeComboBox.clear()
            self.chartTypeComboBox.addItems(["Bar Chart", "Line Chart"])

        self.df['dyad'] = self.df['dyad'].astype(str)
        #dyads = sorted(self.df['dyad'].dropna().unique())

        import re
        dyads = sorted(
            self.df['dyad'].dropna().unique(),
            key=lambda d: int(re.search(r'\d+', d).group()) if re.search(r'\d+', d) else float('inf')
        )

        groups = sorted(self.df['group'].dropna().unique())
        conditions = sorted(self.df['condition'].dropna().unique())

        self.dyad_to_group = dict(self.df[['dyad', 'group']].dropna().values)

        for group in groups:
            item = QtWidgets.QListWidgetItem(group)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.groupListWidget.addItem(item)

        for dyad in dyads:
            item = QtWidgets.QListWidgetItem(dyad)
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.dyadListWidget.addItem(item)

        for condition in conditions:
            item = QtWidgets.QListWidgetItem(condition)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.conditionListWidget.addItem(item)

        self.groupListWidget.itemChanged.connect(self.update_dyad_checkboxes_by_group_selection)

        excluded = {'group', 'condition', 'dyad'}
        node_columns = [col for col in self.df.columns if col not in excluded]
        for col in node_columns:
            item = QtWidgets.QListWidgetItem(col)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.nodeListWidget.addItem(item)

    def update_dyad_checkboxes_by_group_selection(self):
        selected_groups = {
            self.groupListWidget.item(i).text()
            for i in range(self.groupListWidget.count())
            if self.groupListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked
        }

        for i in range(self.dyadListWidget.count()):
            item = self.dyadListWidget.item(i)
            dyad = item.text()
            group = self.dyad_to_group.get(dyad)
            if group in selected_groups:
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def compare_node_strength(self):
        selected_groups = [self.groupListWidget.item(i).text() for i in range(self.groupListWidget.count())
                           if self.groupListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        selected_conditions = [self.conditionListWidget.item(i).text() for i in range(self.conditionListWidget.count())
                               if self.conditionListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        selected_dyads = [self.dyadListWidget.item(i).text() for i in range(self.dyadListWidget.count())
                          if self.dyadListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        selected_nodes = [self.nodeListWidget.item(i).text() for i in range(self.nodeListWidget.count())
                          if self.nodeListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]

        if not selected_groups or not selected_conditions or not selected_dyads or not selected_nodes:
            QMessageBox.warning(self, "Incomplete Selection", "Please select at least one from each list.")
            return

        filtered = self.df[
            self.df['group'].isin(selected_groups) &
            self.df['condition'].isin(selected_conditions) &
            self.df['dyad'].isin(selected_dyads)
        ]

        if filtered.empty:
            QMessageBox.information(self, "No Data", "No matching data found.")
            return

        self.plot_node_strengths(filtered, selected_nodes)

        self.exportButton.setEnabled(True)

    def plot_node_strengths(self, data, selected_nodes):
        """Plot the selected local graph metric across nodes for the chosen dyads and conditions.
           Each dyad’s data is plotted, showing node-wise metric values for comparison."""
        chart_type = self.chartTypeComboBox.currentText() if self.chartTypeComboBox else "Bar Chart"

        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        try:
            if chart_type == "Bar Chart":
                # Melt into long format
                melted = data[["condition", "dyad"] + selected_nodes].melt(
                    id_vars=["condition", "dyad"],
                    value_vars=selected_nodes,
                    var_name="node",
                    value_name="value"
                )

                # Group by node and condition
                summary = melted.groupby(["node", "condition"]).agg(
                    mean=("value", "mean"),
                    std=("value", "std")
                ).reset_index()

                # Compute average across all nodes per condition
                avg_summary = melted.groupby("condition").agg(
                    mean=("value", "mean"),
                    std=("value", "std")
                ).reset_index()
                avg_summary["node"] = "Average"

                # Append average group as another "node"
                full_summary = pd.concat([summary, avg_summary], ignore_index=True)

                nodes = sorted(summary["node"].unique()) + ["Average"]
                conditions = sorted(summary["condition"].unique())
                bar_width = 0.8 / len(conditions)
                x = np.arange(len(nodes))

                for i, condition in enumerate(conditions):
                    means = [
                        full_summary[(full_summary["node"] == node) & (full_summary["condition"] == condition)][
                            "mean"].values[0]
                        if not full_summary[
                            (full_summary["node"] == node) & (full_summary["condition"] == condition)].empty else 0
                        for node in nodes
                    ]
                    stds = [
                        full_summary[(full_summary["node"] == node) & (full_summary["condition"] == condition)][
                            "std"].values[0]
                        if not full_summary[
                            (full_summary["node"] == node) & (full_summary["condition"] == condition)].empty else 0
                        for node in nodes
                    ]
                    ax.bar(x + i * bar_width, means, bar_width, yerr=stds, capsize=5, label=condition)

                ax.set_xticks(x + bar_width * (len(conditions) - 1) / 2)
                ax.set_xticklabels(nodes, rotation=45, ha="right")




            elif chart_type == "Line Chart":
                # Group by condition, plot line per node
                grouped = data.groupby("condition")
                for node in selected_nodes:
                    means = grouped[node].mean()
                    stds = grouped[node].std()
                    ax.errorbar(means.index, means.values, yerr=stds.values, marker='o', capsize=5, label=node)

            else:
                QMessageBox.warning(self, "Chart Error", f"Chart type '{chart_type}' is not supported.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", str(e))
            return

        metric_name = self.metricComboBox.currentText() if self.metricComboBox else "Node Strengths"
        ax.set_title(f"{metric_name} Comparison")
        ax.set_xlabel("Node" if chart_type == "Bar Chart" else "Condition")
        ax.set_ylabel(metric_name)
        if chart_type == "Line Chart":
            ax.legend()
        ax.grid(True)

        layout = self.plotWidget.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(self.plotWidget)
            self.plotWidget.setLayout(layout)
        else:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

        layout.addWidget(canvas)

        self.current_plot_figure = fig  # Store the figure for later
        self.openPlotImageButton.setEnabled(True)  # Enable after plot

        if self.plotLabel:
            self.plotLabel.hide()

    def select_all_dyads(self):
        for i in range(self.dyadListWidget.count()):
            item = self.dyadListWidget.item(i)
            if item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(QtCore.Qt.CheckState.Checked)

    def clear_dyads_selection(self):
        for i in range(self.dyadListWidget.count()):
            item = self.dyadListWidget.item(i)
            if item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def select_all_nodes(self):
        for i in range(self.nodeListWidget.count()):
            item = self.nodeListWidget.item(i)
            item.setCheckState(QtCore.Qt.CheckState.Checked)

    def clear_nodes_selection(self):
        for i in range(self.nodeListWidget.count()):
            item = self.nodeListWidget.item(i)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def open_csv_file(self):
        # Determine which file to open based on selected metric
        selected_metric = self.metricComboBox.currentText() if hasattr(self, 'metricComboBox') else "Node Strength"

        filename_map = {
            "Node Strength": "node_strengths_matrix.csv",
            "Local Efficiency": "local_efficiency_matrix.csv"
        }

        selected_filename = filename_map.get(selected_metric, "node_strengths_matrix.csv")

        # Find the folder path based on the originally loaded CSV
        folder = os.path.dirname(self.csv_path)
        full_path = os.path.join(folder, selected_filename)

        if os.path.exists(full_path):
            if platform.system() == "Windows":
                os.startfile(full_path)
            elif platform.system() == "Darwin":
                subprocess.call(["open", full_path])
            else:
                subprocess.call(["xdg-open", full_path])
        else:
            QMessageBox.warning(self, "Not Found", f"CSV file '{selected_filename}' not found.")

    def open_plot_as_image(self):
        if not hasattr(self, "current_plot_figure"):
            QMessageBox.warning(self, "No Plot", "No plot is currently available.")
            return

        import tempfile
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            self.current_plot_figure.savefig(tmp_file.name, bbox_inches='tight')
            tmp_file_path = tmp_file.name

        QDesktopServices.openUrl(QUrl.fromLocalFile(tmp_file_path))

    def go_back(self):
        self.close()
        if self.parent_window:
            self.parent_window.show()

    def show_help_dialog(self):
            # Create a custom QDialog
            about_dialog = QtWidgets.QDialog(self)
            about_dialog.setWindowTitle("Local Measures Comparison - Help")
            about_dialog.setFixedSize(800, 800)

            # Set custom background color here
            about_dialog.setStyleSheet("background-color: #F1F5F9;")  # Light gray-blue

            # Create and style layout and label
            layout = QtWidgets.QVBoxLayout(about_dialog)

            label = QtWidgets.QLabel(
                "                                     🧠 Local Measures Comparison – Help Guide\n\n"
    "This screen allows you to compare local network measures (Node Strength or Local Efficiency) "
    "across dyads, conditions, and nodes.\n\n"
    "✅ How to Use:\n"
    "1. Select Groups: Choose one or more participant groups.\n"
    "2. Select Conditions: Select conditions you'd like to compare.\n"
    "3. Select Dyads: Based on selected groups, check dyads to include.\n"
    "4. Select Nodes: Choose specific brain regions (nodes) for comparison.\n"
    "5. Choose Chart Type: Bar Chart or Line Chart.\n"
    "6. Click \"Compare\" to visualize the data.\n\n"
    "📊 Recommended Analysis Combinations:\n"
    "⭐ Node activity across conditions:\n"
    "  Select one node, multiple conditions, and all dyads to see how a specific brain region behaves in different scenarios.\n\n"
    "⭐ Group-level differences:\n"
    "  Select a single condition and node, then compare across groups/dyads to assess group-specific patterns.\n\n"
    "⭐ All nodes averaged per condition:\n"
    "  Select multiple nodes and multiple conditions to compare overall activity per condition, summarized as \"Average\" in the bar chart.\n\n"
    "Use \"Select All\"/\"Clear\" for quick node and dyad selection.\n"
    "Click \"Export\" to save the figure, summary, and data for reporting."
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

    def metric_changed(self):
        selected_metric = self.metricComboBox.currentText()

        filename_map = {
            "Node Strength": "node_strengths_matrix.csv",
            "Local Efficiency": "local_efficiency_matrix.csv"
        }

        csv_file = filename_map.get(selected_metric)
        if not csv_file:
            QMessageBox.warning(self, "Unsupported Metric", f"Metric '{selected_metric}' is not supported.")
            return

        # 🔧 Derive full path from current csv_path folder
        folder = os.path.dirname(self.csv_path)
        full_path = os.path.join(folder, csv_file)

        if not os.path.exists(full_path):
            QMessageBox.critical(self, "File Missing", f"{full_path} not found.")
            return

        try:
            self.csv_path = full_path
            self.df = pd.read_csv(full_path)
            self.populate_fields()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load {csv_file}:\n{str(e)}")

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
        export_dir = os.path.join(base_folder, "Results", "Local Measures")
        os.makedirs(export_dir, exist_ok=True)

        # Current selections
        selected_groups = [self.groupListWidget.item(i).text() for i in range(self.groupListWidget.count())
                           if self.groupListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        selected_conditions = [self.conditionListWidget.item(i).text() for i in range(self.conditionListWidget.count())
                               if self.conditionListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        selected_dyads = [self.dyadListWidget.item(i).text() for i in range(self.dyadListWidget.count())
                          if self.dyadListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        selected_nodes = [self.nodeListWidget.item(i).text() for i in range(self.nodeListWidget.count())
                          if self.nodeListWidget.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        selected_metric = self.metricComboBox.currentText()
        chart_type = self.chartTypeComboBox.currentText()

        filtered = self.df[
            self.df['group'].isin(selected_groups) &
            self.df['condition'].isin(selected_conditions) &
            self.df['dyad'].isin(selected_dyads)
            ]

        if filtered.empty:
            QMessageBox.warning(self, "No Data", "Nothing to export. Please review your selections.")
            return

        try:
            # Create a unique export subfolder using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dedicated_folder = os.path.join(export_dir, f"Export_{timestamp}")
            os.makedirs(dedicated_folder, exist_ok=True)

            # Save filtered CSV
            filename_base = f"{selected_metric.replace(' ', '_')}_{chart_type.replace(' ', '_')}".lower()
            csv_path = os.path.join(dedicated_folder, f"{filename_base}.csv")
            filtered_cols = ['group', 'condition', 'dyad'] + selected_nodes
            filtered[filtered_cols].to_csv(csv_path, index=False)

            # Save plot
            img_path = os.path.join(dedicated_folder, f"{filename_base}.png")
            self.current_plot_figure.savefig(img_path, bbox_inches='tight')

            # Auto-interpret
            interpretation = self._build_interpretation_local(
                filtered, selected_nodes, chart_type, selected_metric
            )

            # Save summary text
            summary_path = os.path.join(dedicated_folder, "summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("Export Summary\n")
                f.write("==============\n\n")
                f.write(f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Selected Metric: {selected_metric}\n")
                f.write(f"Chart Type: {chart_type}\n\n")
                f.write(f"Selected Groups: {', '.join(selected_groups)}\n")
                f.write(f"Selected Conditions: {', '.join(selected_conditions)}\n")
                f.write(f"Selected Dyads: {', '.join(selected_dyads)}\n")
                f.write(f"Selected Nodes: {', '.join(selected_nodes)}\n\n")
                f.write("Files:\n")
                f.write(f"- {os.path.basename(csv_path)}\n")
                f.write(f"- {os.path.basename(img_path)}\n\n")
                f.write("Interpretation:\n")
                f.write(interpretation + "\n")

            # Save PDF
            pdf_path = os.path.join(dedicated_folder, f"{filename_base}.pdf")
            selections = {
                "Metric": selected_metric,
                "Chart Type": chart_type,
                "Groups": ", ".join(selected_groups),
                "Conditions": ", ".join(selected_conditions),
                "Dyads": ", ".join(selected_dyads),
                "Nodes": ", ".join(selected_nodes),
            }
            self._export_pdf(pdf_path, "Local Measures Export", selections, img_path, interpretation)

            QMessageBox.information(self, "Export Complete",
                                    f"CSV, image, text, and PDF exported to:\n{dedicated_folder}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred during export:\n{e}")

    def generate_export_pdf_report(self, pdf_path, metric, chart_type, selected_groups,
                                   selected_conditions, selected_dyads, selected_nodes,
                                   data_df, plot_img_path, summary_txt_path=None):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"{metric} Comparison Report", ln=True, align='C')

        pdf.ln(10)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 10, f"Chart Type: {chart_type}", ln=True)
        pdf.cell(0, 10, f"Selected Groups: {', '.join(selected_groups)}", ln=True)
        pdf.cell(0, 10, f"Selected Conditions: {', '.join(selected_conditions)}", ln=True)
        pdf.cell(0, 10, f"Selected Dyads: {', '.join(selected_dyads)}", ln=True)
        pdf.cell(0, 10, f"Selected Nodes: {', '.join(selected_nodes)}", ln=True)

        pdf.ln(5)

        # Table (summary of data)
        if not data_df.empty:
            pdf.set_font("Arial", 'B', 10)
            col_names = data_df.columns.tolist()[:6]  # Only first 6 columns
            col_width = pdf.w / len(col_names) - 1
            for col in col_names:
                pdf.cell(col_width, 8, str(col), border=1)
            pdf.ln()

            pdf.set_font("Arial", '', 9)
            for idx, row in data_df.iterrows():
                for col in col_names:
                    val = str(row[col])[:15]
                    pdf.cell(col_width, 8, val, border=1)
                pdf.ln()
                if idx > 20:  # Limit rows to avoid overflow
                    pdf.cell(0, 8, "... (truncated)", ln=True)
                    break

        # Plot image
        if os.path.exists(plot_img_path):
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Comparison Plot:", ln=True)
            pdf.image(plot_img_path, x=None, w=pdf.w - 30)

        # Summary text (optional)
        if summary_txt_path and os.path.exists(summary_txt_path):
            with open(summary_txt_path, "r") as f:
                text = f.read()

            pdf.add_page()
            pdf.set_font("Arial", '', 10)
            pdf.multi_cell(0, 8, text)

        # Save PDF
        pdf.output(pdf_path)

    def _build_interpretation_local(self, filtered: pd.DataFrame, selected_nodes: list,
                                    chart_type: str, metric_name: str) -> str:
        lines = []
        # Choose the first node in list to comment on (or average)
        if not selected_nodes:
            return "No nodes selected; cannot interpret."

        # Mean per condition on all selected nodes
        node_means = filtered[selected_nodes].mean(axis=1)
        df2 = pd.DataFrame({"condition": filtered["condition"], "value": node_means})
        means = df2.groupby("condition")["value"].mean().sort_values(ascending=False)

        top_condition, top_val = means.index[0], means.iloc[0]
        lines.append(f"{metric_name} is highest on average in '{top_condition}' ({top_val:.3f}).")

        stds = df2.groupby("condition")["value"].std().sort_values(ascending=False)
        if not stds.empty and not np.isnan(stds.iloc[0]):
            lines.append(f"The greatest variability appears in '{stds.index[0]}' (STD={stds.iloc[0]:.3f}).")

        ordered = df2.groupby("condition")["value"].mean().reindex(sorted(df2["condition"].unique()))
        if len(ordered) >= 2:
            trend = "increasing" if ordered.iloc[-1] > ordered.iloc[0] else "decreasing"
            lines.append(f"A generally {trend} trend is observed from {ordered.index[0]} to {ordered.index[-1]}.")

        # Nodes that stand out
        per_node_means = filtered[selected_nodes].mean().sort_values(ascending=False)
        if len(per_node_means) > 0:
            lines.append(
                f"Node '{per_node_means.index[0]}' shows the highest average {metric_name} among selected nodes.")

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
        story.append(Image(plot_image_path, width=480, height=300))
        story.append(Spacer(1, 12))

        story.append(Paragraph("<b>Auto-generated Interpretation</b>", styles["Heading3"]))
        story.append(Paragraph(interpretation.replace("\n", "<br/>"), styles["BodyText"]))

        doc.build(story)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LocalMeasureComparisonWindow("node_strengths_matrix.csv")
    window.show()
    sys.exit(app.exec())
