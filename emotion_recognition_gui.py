import sys
import os
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTextEdit,
                             QProgressBar, QGroupBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont

# Import your model and dataloader
from FourierGNNmodel import DialogueGCNModel
from dataloader import IEMOCAPDataset

class InferenceWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model_path, data_path, is_video=False):
        super().__init__()
        self.model_path = model_path
        self.data_path = data_path
        self.is_video = is_video

    def run(self):
        try:
            self.progress.emit(10)

            # Load model
            model = DialogueGCNModel(
                base_model='LSTM',
                D_m=712,  # IEMOCAP dimensions: 100+100+512
                D_e=100,
                graph_hidden_size=100,
                n_speakers=2,
                max_seq_len=200,
                window_past=10,
                window_future=10,
                n_classes=6,
                dropout=0.5,
                nodal_attention=False,
                no_cuda=False,
                graph_type='relation',
                use_topic=False,
                alpha=0.2,
                multiheads=6,
                graph_construct='full',
                use_residue=True,
                D_m_v=512,
                D_m_a=100,
                modals='avl',
                att_type='concat',
                av_using_lstm=True,
                dataset='IEMOCAP',
                use_speaker=True,
                use_modal=True
            )

            if torch.cuda.is_available():
                model.load_state_dict(torch.load(self.model_path))
                model.cuda()
            else:
                model.load_state_dict(torch.load(self.model_path, map_location='cpu'))

            model.eval()
            self.progress.emit(30)

            # For demo purposes, load a sample from the dataset
            # In a real application, you'd process the specific input file
            dataset = IEMOCAPDataset()
            # Get first test sample as example
            sample_idx = len(dataset.train_data) + len(dataset.valid_data)  # Start of test data
            if sample_idx >= len(dataset.data):
                sample_idx = 0  # Fallback to first sample

            data = dataset[sample_idx]
            self.progress.emit(60)

            # Perform inference
            with torch.no_grad():
                textf, acouf, visuf, qmask, umask, label = [d.cuda() if torch.cuda.is_available() else d for d in data[:-1]]
                lengths = torch.sum(umask, dim=1).int().tolist()

                log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
                kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob, e_i, e_n, e_t, e_l = model(textf.unsqueeze(0), acouf.unsqueeze(0), visuf.unsqueeze(0), qmask.unsqueeze(0), umask.unsqueeze(0), lengths)

                # Get probabilities for the sequence
                all_prob_flat = all_prob.view(-1, all_prob.size(-1))
                probs = torch.softmax(all_prob_flat, dim=-1).mean(dim=0).cpu().numpy()  # Average over sequence

            self.progress.emit(100)

            # Emotion labels for IEMOCAP
            emotions = ['Angry', 'Excited', 'Frustrated', 'Happy', 'Neutral', 'Sad']
            results = {emotions[i]: float(probs[i] * 100) for i in range(len(emotions))}

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

class EmotionRecognitionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Emotion Recognition in Dialogue')
        self.setGeometry(300, 300, 600, 500)

        layout = QVBoxLayout()

        # File selection group
        file_group = QGroupBox("Input Selection")
        file_layout = QVBoxLayout()

        # Input type selection
        type_layout = QHBoxLayout()
        self.input_type_group = QButtonGroup()
        self.radio_pkl = QRadioButton("Preprocessed .pkl file")
        self.radio_pkl.setChecked(True)
        self.radio_video = QRadioButton("Video file")
        # self.radio_video.setEnabled(False)
        self.input_type_group.addButton(self.radio_pkl)
        self.input_type_group.addButton(self.radio_video)
        type_layout.addWidget(self.radio_pkl)
        type_layout.addWidget(self.radio_video)
        file_layout.addLayout(type_layout)

        # File selection
        file_select_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        file_select_layout.addWidget(self.file_label)
        file_select_layout.addWidget(self.file_button)
        file_layout.addLayout(file_select_layout)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout()
        self.model_label = QLabel("No model selected")
        self.model_button = QPushButton("Select Model")
        self.model_button.clicked.connect(self.select_model)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_button)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Run button
        self.run_button = QPushButton("Run Emotion Recognition")
        self.run_button.clicked.connect(self.run_inference)
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Results display
        results_group = QGroupBox("Recognition Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 10))
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.setLayout(layout)

        self.selected_file = None
        self.selected_model = None

    def select_file(self):
        if self.radio_pkl.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Preprocessed File", "", "PKL files (*.pkl)")
        else:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video files (*.mp4 *.avi *.mov)")

        if file_path:
            self.selected_file = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.check_run_enabled()

    def select_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch models (*.pth)")
        if model_path:
            self.selected_model = model_path
            self.model_label.setText(os.path.basename(model_path))
            self.check_run_enabled()

    def check_run_enabled(self):
        self.run_button.setEnabled(self.selected_file is not None and self.selected_model is not None)

    def run_inference(self):
        self.progress_bar.setValue(0)
        self.results_text.clear()
        self.results_text.append("Starting emotion recognition...")

        is_video = self.radio_video.isChecked()
        self.worker = InferenceWorker(self.selected_model, self.selected_file, is_video)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.display_error)
        self.worker.start()

    def display_results(self, results):
        self.results_text.clear()
        self.results_text.append("Emotion Recognition Results:\n")
        self.results_text.append("-" * 40)

        # Sort by probability
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        for emotion, prob in sorted_results:
            self.results_text.append(f"{emotion}: {prob*100:.2f}%")

        self.results_text.append("\n" + "=" * 40)
        self.results_text.append("Analysis complete!")

    def display_error(self, error_msg):
        self.results_text.clear()
        self.results_text.append("Error occurred:")
        self.results_text.append(error_msg)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EmotionRecognitionGUI()
    gui.show()
    sys.exit(app.exec_())