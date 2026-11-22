import sys
import json
import os
import time
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QMessageBox, QHeaderView, QTableView, QGroupBox,
    QScrollArea, QFrame, QFileDialog, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, QAbstractTableModel
from huggingface_hub import InferenceClient
from datasets import load_dataset

# ==========================================
# GLOBAL DARK THEME STYLESHEET
# ==========================================
STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 14px;
}
QTabWidget::pane {
    border: 1px solid #313244;
    background: #181825;
}
QTabBar::tab {
    background: #181825;
    color: #a6adc8;
    padding: 8px 16px;
    border: 1px solid #313244;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    background: #313244;
    color: #cdd6f4;
    margin-bottom: -1px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px 8px 8px 8px;
    background-color: #181825;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #89b4fa;
    font-weight: bold;
}
QPushButton {
    background-color: #89b4fa;
    color: #11111b;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #b4befe;
}
QPushButton:pressed {
    background-color: #74c7ec;
}
QPushButton:disabled {
    background-color: #45475a;
    color: #a6adc8;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    padding: 6px;
    border-radius: 4px;
}
QTableView {
    background-color: #181825;
    gridline-color: #313244;
    border: 1px solid #313244;
    border-radius: 4px;
}
QHeaderView::section {
    background-color: #313244;
    padding: 6px;
    border: 1px solid #181825;
    color: #cdd6f4;
}
QLabel {
    color: #cdd6f4;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    background-color: #313244;
}
QProgressBar::chunk {
    background-color: #a6e3a1;
    border-radius: 4px;
}
"""

# ==========================================
# THREADS FOR ASYNC OPERATIONS
# ==========================================

class DatasetLoaderThread(QThread):
    data_loaded = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name

    def run(self):
        try:
            dataset = load_dataset(self.dataset_name, split="train")
            sample = dataset.select(range(min(5, len(dataset))))
            self.data_loaded.emit((dataset, sample.to_pandas()))
        except Exception as e:
            self.error_occurred.emit(str(e))

class InferenceThread(QThread):
    response_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, token, model_id, prompt, system_prompt):
        super().__init__()
        self.token = token
        self.model_id = model_id
        self.prompt = prompt
        self.system_prompt = system_prompt

    def run(self):
        try:
            client = InferenceClient(model=self.model_id, token=self.token)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.prompt}
            ]
            response = client.chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.1,
                top_p=0.9
            )
            self.response_ready.emit(response.choices[0].message.content)
        except Exception as e:
            self.error_occurred.emit(str(e))

class TrainingSimThread(QThread):
    progress_update = Signal(int)
    log_update = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm

    def run(self):
        try:
            self.log_update.emit(f"Initializing {self.algorithm} training environment...")
            time.sleep(1)
            
            for i in range(1, 101):
                self.progress_update.emit(i)
                if i == 10:
                    self.log_update.emit("Loading dataset and initializing model weights...")
                elif i == 30:
                    self.log_update.emit("Forward pass and computing loss...")
                elif i == 50:
                    self.log_update.emit("Backward pass and optimizing parameters...")
                elif i == 80:
                    self.log_update.emit("Evaluating model on validation set...")
                elif i == 100:
                    self.log_update.emit("Training simulation completed successfully! Model saved to ./checkpoints")
                time.sleep(0.05)
        except Exception as e:
            self.error_occurred.emit(str(e))

# ==========================================
# CUSTOM TABLE MODEL
# ==========================================

class PandasModel(QAbstractTableModel):
    def __init__(self, df):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            if isinstance(value, str) and len(value) > 100:
                return value[:100] + "..."
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            if orientation == Qt.Vertical:
                return str(self._df.index[section])
        return None

# ==========================================
# HELPER UI FUNCTIONS
# ==========================================

def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()

# ==========================================
# TAB 1: DATASET PREPARATION
# ==========================================

class DatasetPreparationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        load_layout = QHBoxLayout()
        load_layout.addWidget(QLabel("HF Dataset Name:"))
        self.dataset_input = QLineEdit("openai_humaneval")
        load_layout.addWidget(self.dataset_input)
        
        load_btn = QPushButton("Load Dataset")
        load_btn.clicked.connect(self.load_dataset)
        load_layout.addWidget(load_btn)
        layout.addLayout(load_layout)

        self.table_view = QTableView()
        self.table_view.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QLabel("Dataset Preview (First 5 rows):"))
        layout.addWidget(self.table_view)

        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(QLabel("Prompt Col:"))
        self.prompt_col_input = QLineEdit("prompt")
        fmt_layout.addWidget(QLabel("Completion Col:"))
        self.completion_col_input = QLineEdit("canonical_solution")
        
        self.output_format = QComboBox()
        self.output_format.addItems(["Instruction JSONL (SFT)", "Preference Pairs (DPO)", "Process Annotations (PRM)", "GRPO Trajectories"])
        fmt_layout.addWidget(self.output_format)
        
        export_btn = QPushButton("Export Formatted Dataset")
        export_btn.clicked.connect(self.export_dataset)
        fmt_layout.addWidget(export_btn)
        layout.addLayout(fmt_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        layout.addWidget(QLabel("Processing Log:"))
        layout.addWidget(self.log_text)

    def load_dataset(self):
        dataset_name = self.dataset_input.text().strip()
        if not dataset_name: 
            return
        self.log_text.append(f"Loading {dataset_name}...")
        self.loader_thread = DatasetLoaderThread(dataset_name)
        self.loader_thread.data_loaded.connect(self.on_dataset_loaded)
        self.loader_thread.error_occurred.connect(self.on_error)
        self.loader_thread.start()

    def on_dataset_loaded(self, data):
        self.dataset, df = data
        model = PandasModel(df)
        self.table_view.setModel(model)
        self.table_view.resizeColumnsToContents()
        cols = self.dataset.column_names
        self.prompt_col_input.setText(cols[0])
        self.completion_col_input.setText(cols[1] if len(cols) > 1 else cols[0])
        self.log_text.append(f"<span style='color:#a6e3a1'>Successfully loaded {len(self.dataset)} rows.</span>")

    def on_error(self, error):
        self.log_text.append(f"<span style='color:#f38ba8'>Error: {error}</span>")

    def export_dataset(self):
        if not self.dataset:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return
        prompt_col = self.prompt_col_input.text().strip()
        completion_col = self.completion_col_input.text().strip()
        
        if prompt_col not in self.dataset.column_names or completion_col not in self.dataset.column_names:
            QMessageBox.warning(self, "Warning", "Invalid column names.")
            return

        default_filename = f"{self.dataset_input.text().replace('/', '_')}_formatted.jsonl"
        filename, _ = QFileDialog.getSaveFileName(self, "Save Formatted Dataset", default_filename, "JSONL Files (*.jsonl)")
        
        if not filename:
            return

        count = 0
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for row in self.dataset:
                    fmt_type = self.output_format.currentText()
                    if "Instruction" in fmt_type:
                        formatted = {
                            "messages": [
                                {"role": "system", "content": "You are an expert AI coding assistant."}, 
                                {"role": "user", "content": row[prompt_col]}, 
                                {"role": "assistant", "content": row[completion_col]}
                            ]
                        }
                    elif "Preference" in fmt_type:
                        # Fallback safely if 'wrong_solution' doesn't exist
                        rejected = row.get("wrong_solution", row.get("rejected", ""))
                        formatted = {"prompt": row[prompt_col], "chosen": row[completion_col], "rejected": rejected}
                    elif "Process" in fmt_type:
                        formatted = {"prompt": row[prompt_col], "steps": [{"text": row[completion_col], "reward": 1.0}], "final_reward": 1.0}
                    elif "GRPO" in fmt_type:
                        formatted = {"prompt": row[prompt_col], "completions": [row[completion_col]], "rewards": [1.0]}
                    f.write(json.dumps(formatted) + '\n')
                    count += 1
            self.log_text.append(f"<span style='color:#a6e3a1'>Exported {count} rows to {filename}</span>")
        except Exception as e:
            self.log_text.append(f"<span style='color:#f38ba8'>Export failed: {str(e)}</span>")

# ==========================================
# TAB 2: ADVANCED MODEL ARCHITECTURE
# ==========================================

class AdvancedArchitectureWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.layout = QVBoxLayout(container)

        self.arch_type = QComboBox()
        self.arch_type.addItems([
            "Dense Transformer (Standard)", "Spatial Transformer (Code Layout-Aware)",
            "Mamba (State Space Model)", "AST-Transformer (Structure-Aware)",
            "GNN-Coder (Graph Neural Network)", "TreeDiff (AST-Guided Diffusion)",
            "Mixture-of-Experts (MoE)", "Long-Context Model (1M+ tokens)"
        ])
        self.arch_type.currentTextChanged.connect(self.on_arch_changed)
        self.layout.addWidget(QLabel("Select Model Architecture:"))
        self.layout.addWidget(self.arch_type)

        self.form_layout = QFormLayout()
        self.layout.addLayout(self.form_layout)
        
        self.diagram_label = QLabel()
        self.diagram_label.setWordWrap(True)
        self.diagram_label.setStyleSheet("color: #bac2de; font-style: italic; padding: 10px; background: #181825; border-radius: 5px;")
        self.layout.addWidget(self.diagram_label)
        self.layout.addStretch()

        scroll.setWidget(container)
        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(scroll)
        self.on_arch_changed(self.arch_type.currentText())

    def on_arch_changed(self, text):
        clear_layout(self.form_layout)
        if "Mamba" in text: self.add_mamba_params()
        elif "Spatial" in text: self.add_spatial_params()
        elif "AST-Transformer" in text: self.add_ast_params()
        elif "GNN" in text: self.add_gnn_params()
        elif "TreeDiff" in text: self.add_treediff_params()
        elif "MoE" in text: self.add_moe_params()
        elif "Long-Context" in text: self.add_long_context_params()
        else: self.add_dense_params()

    def add_dense_params(self):
        self.form_layout.addRow("Hidden Dimension:", QLineEdit("4096"))
        self.form_layout.addRow("Num Layers:", QLineEdit("32"))
        self.diagram_label.setText("Standard Transformer: O(N²) attention complexity. Baseline for code generation.")

    def add_mamba_params(self):
        self.form_layout.addRow("State Dimension:", QLineEdit("16"))
        self.form_layout.addRow("Model Dimension:", QLineEdit("768"))
        self.form_layout.addRow("Expand Factor:", QLineEdit("2"))
        combo = QComboBox(); combo.addItems(["HiPPO-LegS", "S4D-Lin", "S4D-Real"])
        self.form_layout.addRow("Init Method:", combo)
        self.diagram_label.setText("Mamba SSM: Linear-time sequence modeling O(1) inference. Selective mechanism enables content-aware representations.")

    def add_spatial_params(self):
        self.form_layout.addRow("Spatial Dimensions:", QLineEdit("2D (Row, Column)"))
        self.form_layout.addRow("Patch Size:", QLineEdit("16x16"))
        self.diagram_label.setText("Spatial Transformer: Processes code as 2D layout (rows, cols). Captures indentation and hierarchical structure.")

    def add_ast_params(self):
        combo1 = QComboBox(); combo1.addItems(["Pre-order", "Post-order", "Sibling-based"])
        self.form_layout.addRow("AST Traversal:", combo1)
        self.form_layout.addRow("Node Embedding Size:", QLineEdit("768"))
        combo2 = QComboBox(); combo2.addItems(["Ancestor-Descendant", "Sibling", "Full"])
        self.form_layout.addRow("Attention Masking:", combo2)
        self.diagram_label.setText("AST-Transformer: Directly encodes ASTs. Reduces complexity to O(N·K). Captures syntactic relationships.")

    def add_gnn_params(self):
        self.form_layout.addRow("GNN Layers:", QLineEdit("4"))
        self.form_layout.addRow("Hidden Dimension:", QLineEdit("256"))
        combo = QComboBox(); combo.addItems(["Mean", "Max", "AST-Specific"])
        self.form_layout.addRow("Graph Pooling:", combo)
        self.diagram_label.setText("GNN-Coder: Uses Graph Neural Networks on ASTs. Captures structural dependencies in code.")

    def add_treediff_params(self):
        self.form_layout.addRow("Diffusion Steps:", QLineEdit("100"))
        combo = QComboBox(); combo.addItems(["Random Token", "AST Subtree", "Hybrid"])
        self.form_layout.addRow("Corruption Strategy:", combo)
        self.diagram_label.setText("TreeDiff: AST-guided diffusion model. Syntax-aware corruption preserves grammatical boundaries.")

    def add_moe_params(self):
        self.form_layout.addRow("Number of Experts:", QLineEdit("8"))
        self.form_layout.addRow("Expert Capacity:", QLineEdit("2.0"))
        combo = QComboBox(); combo.addItems(["Top-2", "MaxScore", "Adaptive"])
        self.form_layout.addRow("Routing Strategy:", combo)
        self.diagram_label.setText("MoE Architecture: Conditional computation reduces FLOPs by 60%. DeepSeek-V3 style with 256 experts.")

    def add_long_context_params(self):
        self.form_layout.addRow("Max Context Length:", QLineEdit("1,000,000"))
        combo = QComboBox(); combo.addItems(["Causal Retrieval", "Ring Attention", "Sparse"])
        self.form_layout.addRow("Attention Mechanism:", combo)
        self.diagram_label.setText("Long-Context: Processes entire code repositories (1M+ tokens). 1000x extrapolation capability.")

# ==========================================
# TAB 3: LLM ATTRIBUTES & HYPERPARAMETERS
# ==========================================

class LLMAttributesWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        sizing_group = QGroupBox("Core Architecture Sizing")
        sizing_layout = QFormLayout()
        self.hidden_dim = QSpinBox(); self.hidden_dim.setRange(256, 32768); self.hidden_dim.setValue(4096)
        sizing_layout.addRow("Hidden Dimension:", self.hidden_dim)
        self.num_layers = QSpinBox(); self.num_layers.setRange(4, 128); self.num_layers.setValue(32)
        sizing_layout.addRow("Number of Layers:", self.num_layers)
        self.num_heads = QSpinBox(); self.num_heads.setRange(4, 64); self.num_heads.setValue(32)
        sizing_layout.addRow("Attention Heads:", self.num_heads)
        self.kv_heads = QSpinBox(); self.kv_heads.setRange(1, 64); self.kv_heads.setValue(8)
        sizing_layout.addRow("KV Heads (GQA):", self.kv_heads)
        sizing_group.setLayout(sizing_layout)
        layout.addWidget(sizing_group)

        pos_group = QGroupBox("Positional Encoding & Context")
        pos_layout = QFormLayout()
        self.pos_encoding = QComboBox()
        self.pos_encoding.addItems(["RoPE", "ALiBi", "2D Spatial", "AST-Depth"])
        pos_layout.addRow("Position Encoding Type:", self.pos_encoding)
        self.rope_theta = QDoubleSpinBox(); self.rope_theta.setRange(1000.0, 1000000.0); self.rope_theta.setValue(10000.0)
        pos_layout.addRow("RoPE Base Theta:", self.rope_theta)
        self.context_length = QSpinBox(); self.context_length.setRange(512, 10000000); self.context_length.setValue(16384)
        pos_layout.addRow("Max Context Length:", self.context_length)
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)

        token_group = QGroupBox("Tokenization & Code-Specific Attributes")
        token_layout = QFormLayout()
        self.tokenizer_type = QComboBox()
        self.tokenizer_type.addItems(["Byte-Level BPE", "Code-Specific BPE", "Unigram", "AST-Aware"])
        token_layout.addRow("Tokenizer Type:", self.tokenizer_type)
        self.fim_support = QCheckBox("Enable Fill-in-the-Middle (FIM)")
        token_layout.addRow("", self.fim_support)
        self.indent_aware = QCheckBox("Indentation-Aware Embeddings")
        token_layout.addRow("", self.indent_aware)
        token_group.setLayout(token_layout)
        layout.addWidget(token_group)

        decode_group = QGroupBox("Decoding & Code Constraints")
        decode_layout = QFormLayout()
        self.temp = QDoubleSpinBox(); self.temp.setRange(0.0, 2.0); self.temp.setValue(0.2)
        decode_layout.addRow("Temperature:", self.temp)
        self.top_p = QDoubleSpinBox(); self.top_p.setRange(0.0, 1.0); self.top_p.setValue(0.95)
        decode_layout.addRow("Top-p:", self.top_p)
        self.rep_penalty = QDoubleSpinBox(); self.rep_penalty.setRange(1.0, 2.0); self.rep_penalty.setValue(1.1)
        decode_layout.addRow("Repetition Penalty:", self.rep_penalty)
        self.constrained_decoding = QComboBox()
        self.constrained_decoding.addItems(["None", "AST-Constrained", "Grammar-Constrained", "Levenshtein"])
        decode_layout.addRow("Constrained Decoding:", self.constrained_decoding)
        decode_group.setLayout(decode_layout)
        layout.addWidget(decode_group)

        export_btn = QPushButton("Export LLM Configuration to JSON")
        export_btn.clicked.connect(self.export_config)
        layout.addWidget(export_btn)
        layout.addStretch()

        scroll.setWidget(container)
        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(scroll)

    def export_config(self):
        config = {
            "architecture": {"hidden_dim": self.hidden_dim.value(), "num_layers": self.num_layers.value(), "num_heads": self.num_heads.value(), "kv_heads": self.kv_heads.value()},
            "positional_encoding": {"type": self.pos_encoding.currentText(), "rope_theta": self.rope_theta.value(), "max_context": self.context_length.value()},
            "tokenization": {"type": self.tokenizer_type.currentText(), "fim_enabled": self.fim_support.isChecked(), "indent_aware": self.indent_aware.isChecked()},
            "decoding": {"temperature": self.temp.value(), "top_p": self.top_p.value(), "rep_penalty": self.rep_penalty.value(), "constrained": self.constrained_decoding.currentText()}
        }
        filename, _ = QFileDialog.getSaveFileName(self, "Save LLM Config", "llm_config.json", "JSON Files (*.json)")
        if not filename: return
        try:
            with open(filename, 'w') as f: json.dump(config, f, indent=4)
            QMessageBox.information(self, "Success", f"Exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

# ==========================================
# TAB 4: ACTIVATION FUNCTIONS
# ==========================================

class ActivationFunctionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.activation_type = QComboBox()
        self.activation_type.addItems(["ReLU", "GELU", "SwiGLU", "PowLU", "Evolved Activation", "Mish"])
        self.activation_type.currentTextChanged.connect(self.on_activation_changed)
        layout.addWidget(QLabel("Select Activation Function:"))
        layout.addWidget(self.activation_type)

        self.form_layout = QFormLayout()
        layout.addLayout(self.form_layout)

        self.viz_label = QLabel()
        self.viz_label.setWordWrap(True)
        self.viz_label.setStyleSheet("color: #bac2de; font-style: italic; padding: 10px; background: #181825; border-radius: 5px;")
        layout.addWidget(self.viz_label)
        layout.addStretch()
        self.on_activation_changed(self.activation_type.currentText())

    def on_activation_changed(self, text):
        clear_layout(self.form_layout)
        if "SwiGLU" in text:
            self.form_layout.addRow("Swish Beta:", QLineEdit("1.0"))
            combo = QComboBox(); combo.addItems(["Zero", "Identity", "Xavier"])
            self.form_layout.addRow("Gating Init:", combo)
            self.viz_label.setText("SwiGLU: Combines Swish and Gating Linear Unit. Standard in 2026 SOTA LLMs. Formula: SwiGLU(x) = Swish(xW) ⊗ xV")
        elif "PowLU" in text:
            self.form_layout.addRow("Power Law Exponent:", QLineEdit("0.5"))
            self.viz_label.setText("PowLU: Novel power law activation. Addresses gradient vanishing/exploding in deep models.")
        elif "Evolved" in text:
            self.form_layout.addRow("Search Space:", QCheckBox("Include standard activations"))
            self.viz_label.setText("Evolved: LLM-guided search for optimal activations. Discovers novel mathematical forms.")
        else:
            self.viz_label.setText("Standard activation function selected.")

# ==========================================
# TAB 5: ENTROPY REGULARIZATION
# ==========================================

class EntropyRegularizationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.entropy_method = QComboBox()
        self.entropy_method.addItems(["None", "Standard Entropy", "EPO (Entropy-Regularized)", "Adaptive with KL Control", "SVS (Self-Play)"])
        self.entropy_method.currentTextChanged.connect(self.on_method_changed)
        layout.addWidget(QLabel("Entropy Regularization Method:"))
        layout.addWidget(self.entropy_method)

        self.form_layout = QFormLayout()
        layout.addLayout(self.form_layout)

        self.viz_label = QLabel()
        self.viz_label.setWordWrap(True)
        self.viz_label.setStyleSheet("color: #bac2de; font-style: italic; padding: 10px; background: #181825; border-radius: 5px;")
        layout.addWidget(self.viz_label)
        layout.addStretch()
        self.on_method_changed(self.entropy_method.currentText())

    def on_method_changed(self, text):
        clear_layout(self.form_layout)
        if "EPO" in text:
            self.form_layout.addRow("Entropy Coefficient:", QLineEdit("0.01"))
            self.form_layout.addRow("KL Penalty:", QLineEdit("0.1"))
            self.viz_label.setText("EPO: Addresses policy entropy collapse in RL. Balances exploration and exploitation.")
        elif "SVS" in text:
            self.form_layout.addRow("Self-Play Iterations:", QLineEdit("100"))
            self.form_layout.addRow("Entropy Floor:", QLineEdit("0.01"))
            self.viz_label.setText("SVS: Sustains RLVR by preventing entropy collapse. Automatically generates diverse problems.")
        elif "Adaptive" in text:
            self.form_layout.addRow("Target Entropy:", QLineEdit("-1.0"))
            self.viz_label.setText("Adaptive Entropy: Dynamically adjusts entropy bonus based on KL divergence from reference model.")

# ==========================================
# TAB 6: AST GENERATORS
# ==========================================

class ASTGeneratorWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.ast_method = QComboBox()
        self.ast_method.addItems(["Tree-sitter", "ANTLR", "Python ast Module", "Custom AST", "Hybrid (Multiple Parsers)"])
        self.ast_method.currentTextChanged.connect(self.on_method_changed)
        layout.addWidget(QLabel("AST Generation Method:"))
        layout.addWidget(self.ast_method)

        self.form_layout = QFormLayout()
        layout.addLayout(self.form_layout)

        self.viz_label = QLabel()
        self.viz_label.setWordWrap(True)
        self.viz_label.setStyleSheet("color: #bac2de; font-style: italic; padding: 10px; background: #181825; border-radius: 5px;")
        layout.addWidget(self.viz_label)
        layout.addStretch()
        self.on_method_changed(self.ast_method.currentText())

    def on_method_changed(self, text):
        clear_layout(self.form_layout)
        if "Tree-sitter" in text:
            combo = QComboBox(); combo.addItems(["Python", "JS", "Java", "C++", "All"])
            self.form_layout.addRow("Supported Languages:", combo)
            self.form_layout.addRow("Error Recovery:", QCheckBox("Enable robust parsing"))
            self.viz_label.setText("Tree-sitter: Universal parser. Error recovery for incomplete code. Incremental parsing for IDEs.")
        elif "Hybrid" in text:
            combo1 = QComboBox(); combo1.addItems(["Tree-sitter", "ANTLR"])
            self.form_layout.addRow("Primary Parser:", combo1)
            combo2 = QComboBox(); combo2.addItems(["Voting", "Priority", "ML Fusion"])
            self.form_layout.addRow("Consensus Mechanism:", combo2)
            self.viz_label.setText("Hybrid: Combines multiple parsers. Consensus mechanism for conflicting ASTs. >95% parsing accuracy.")

# ==========================================
# TAB 7: TRAINING PIPELINE
# ==========================================

class TrainingPipelineWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        pipeline_label = QLabel("2026 Frontier Training Pipeline:")
        pipeline_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #89b4fa;")
        layout.addWidget(pipeline_label)

        pipeline_diagram = QLabel("1. SFT Cold-Start  →  2. DPO Alignment  →  3. GRPO Reinforcement Learning\n4. PRM Training  →  5. Test-Time Scaling  →  6. Evaluation")
        pipeline_diagram.setStyleSheet("background: #313244; padding: 10px; border-radius: 5px; font-family: monospace;")
        layout.addWidget(pipeline_diagram)

        self.algo_type = QComboBox()
        self.algo_type.addItems(["SFT (Cold Start)", "DPO (Alignment)", "GRPO (RL)", "PRM Training", "Async RL (Long-Horizon)"])
        self.algo_type.currentTextChanged.connect(self.on_algo_changed)
        layout.addWidget(QLabel("Select Training Algorithm:"))
        layout.addWidget(self.algo_type)

        self.form_layout = QFormLayout()
        layout.addLayout(self.form_layout)

        control_layout = QHBoxLayout()
        self.train_button = QPushButton("Start Training Simulation")
        self.train_button.clicked.connect(self.start_training)
        self.export_button = QPushButton("Export Training Config")
        control_layout.addWidget(self.train_button)
        control_layout.addWidget(self.export_button)
        layout.addLayout(control_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(QLabel("Training Log:"))
        layout.addWidget(self.log_text)
        
        self.on_algo_changed(self.algo_type.currentText())

    def on_algo_changed(self, text):
        clear_layout(self.form_layout)
        if "SFT" in text:
            self.form_layout.addRow("Learning Rate:", QLineEdit("2e-5"))
            self.form_layout.addRow("Epochs:", QLineEdit("3"))
        elif "DPO" in text:
            self.form_layout.addRow("Beta:", QLineEdit("0.1"))
            self.form_layout.addRow("Learning Rate:", QLineEdit("5e-7"))
        elif "GRPO" in text:
            self.form_layout.addRow("Group Size:", QLineEdit("8"))
            self.form_layout.addRow("KL Penalty:", QLineEdit("0.1"))
            combo = QComboBox(); combo.addItems(["Group Relative", "Moving Average"])
            self.form_layout.addRow("Advantage Calc:", combo)
        elif "PRM" in text:
            combo1 = QComboBox(); combo1.addItems(["Step-Level", "Outcome-Level"])
            self.form_layout.addRow("Supervision Level:", combo1)
            combo2 = QComboBox(); combo2.addItems(["Execution", "Test Cases", "Static Analysis"])
            self.form_layout.addRow("Feedback Source:", combo2)
        elif "Async" in text:
            self.form_layout.addRow("Rollout Workers:", QLineEdit("16"))
            combo = QComboBox(); combo.addItems(["verl", "OpenRLHF", "DORA"])
            self.form_layout.addRow("Framework:", combo)

    def start_training(self):
        self.train_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        algo = self.algo_type.currentText()
        self.training_thread = TrainingSimThread(algo)
        self.training_thread.progress_update.connect(self.progress_bar.setValue)
        self.training_thread.log_update.connect(self.log_text.append)
        self.training_thread.finished.connect(lambda: self.train_button.setEnabled(True))
        self.training_thread.start()

# ==========================================
# TAB 8: INFERENCE PLAYGROUND
# ==========================================

class InferencePlaygroundWidget(QWidget):
    def __init__(self, get_token_callback):
        super().__init__()
        self.get_token = get_token_callback
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model ID:"))
        self.model_input = QLineEdit("Qwen/Qwen2.5-Coder-7B-Instruct")
        model_layout.addWidget(self.model_input)
        layout.addLayout(model_layout)

        layout.addWidget(QLabel("System Prompt:"))
        self.system_prompt_input = QLineEdit("You are a helpful coding assistant that writes clean, efficient Python code.")
        layout.addWidget(self.system_prompt_input)

        layout.addWidget(QLabel("User Prompt (Coding Task):"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Write a Python function that takes a list of integers and returns the prime numbers...")
        layout.addWidget(self.prompt_input)

        self.generate_btn = QPushButton("Generate Code")
        self.generate_btn.clicked.connect(self.generate_code)
        layout.addWidget(self.generate_btn)

        layout.addWidget(QLabel("Model Output:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

    def generate_code(self):
        token = self.get_token()
        model_id = self.model_input.text().strip()
        prompt = self.prompt_input.toPlainText().strip()
        system_prompt = self.system_prompt_input.text().strip()
        
        if not token:
            QMessageBox.warning(self, "Warning", "Please enter your HF API Token in the top bar.")
            return
        if not prompt: return

        self.output_text.setPlainText("Generating...")
        self.generate_btn.setEnabled(False)
        
        self.inference_thread = InferenceThread(token, model_id, prompt, system_prompt)
        self.inference_thread.response_ready.connect(self.on_response)
        self.inference_thread.error_occurred.connect(self.on_error)
        self.inference_thread.start()

    def on_response(self, response):
        self.output_text.setPlainText(response)
        self.generate_btn.setEnabled(True)

    def on_error(self, error):
        self.output_text.setPlainText(f"Error:\n{error}")
        self.generate_btn.setEnabled(True)

# ==========================================
# MAIN APPLICATION WINDOW
# ==========================================

class SotaCoderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOTA Frontier AI Coding Model Platform - 2026 Edition")
        self.resize(1400, 900)
        
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        header = QLabel("🚀 SOTA Frontier AI Coding Model Platform")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #89b4fa; padding: 10px;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("HF API Token:"))
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)
        self.token_input.setPlaceholderText("hf_...")
        
        # Pre-fill token from environment variables if available
        if os.environ.get("HF_TOKEN"):
            self.token_input.setText(os.environ.get("HF_TOKEN"))
            
        token_layout.addWidget(self.token_input)
        main_layout.addLayout(token_layout)

        self.tabs = QTabWidget()
        self.tabs.addTab(DatasetPreparationWidget(), "1. Dataset Prep")
        self.tabs.addTab(AdvancedArchitectureWidget(), "2. Architecture")
        self.tabs.addTab(LLMAttributesWidget(), "3. LLM Attributes")
        self.tabs.addTab(ActivationFunctionWidget(), "4. Activations")
        self.tabs.addTab(EntropyRegularizationWidget(), "5. Entropy Reg.")
        self.tabs.addTab(ASTGeneratorWidget(), "6. AST Generators")
        self.tabs.addTab(TrainingPipelineWidget(), "7. Training Pipeline")
        self.tabs.addTab(InferencePlaygroundWidget(self.get_token), "8. Playground")
        
        main_layout.addWidget(self.tabs)
        self.setCentralWidget(main_widget)

    def get_token(self):
        return self.token_input.text().strip()

# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = SotaCoderApp()
    window.show()
    sys.exit(app.exec())