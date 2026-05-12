import torch
import numpy as np
import argparse
from FourierGNNmodel import DialogueGCNModel
from dataloader import IEMOCAPDataset

def load_model(model_path):
    """Load the trained model"""
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
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()
    return model

def predict_emotion(model, sample_data):
    """Predict emotions for a single sample"""
    with torch.no_grad():
        textf, visuf, acouf, qmask, umask, label = sample_data[:-1]  # Correct order: text, visual, audio

        # Ensure sequence length doesn't exceed model max_seq_len
        max_len = min(len(textf), 200)

        # Truncate if necessary
        textf = textf[:max_len]
        acouf = acouf[:max_len]
        visuf = visuf[:max_len]
        qmask = qmask[:max_len]
        umask = umask[:max_len]

        # Convert to tensors
        textf = torch.FloatTensor(textf)
        acouf = torch.FloatTensor(acouf)
        visuf = torch.FloatTensor(visuf)
        qmask = torch.FloatTensor(qmask)
        umask = torch.FloatTensor(umask)

        # Concat modalities as in training
        textf = torch.cat([acouf, visuf, textf], dim=-1)  # [seq_len, 712]

        # Add batch dimension - match the shape expected by pad_sequence
        # textf: [batch, seq_len, feature] -> [1, seq_len, feature]
        # qmask: [seq_len, batch, 2] -> [seq_len, 1, 2]
        # umask: [seq_len, batch] -> [seq_len, 1]
        if torch.cuda.is_available():
            textf = textf.unsqueeze(0).cuda()
            acouf = acouf.unsqueeze(0).cuda()
            visuf = visuf.unsqueeze(0).cuda()
            qmask = qmask.unsqueeze(1).cuda()  # [seq_len, 1, 2]
            umask = umask.unsqueeze(1).cuda()  # [seq_len, 1]
        else:
            textf = textf.unsqueeze(0)
            acouf = acouf.unsqueeze(0)
            visuf = visuf.unsqueeze(0)
            qmask = qmask.unsqueeze(1)  # [seq_len, 1, 2]
            umask = umask.unsqueeze(1)  # [seq_len, 1]

        lengths = torch.sum(umask, dim=0).int().tolist()  # sum over seq_len dim

        # Forward pass
        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
        kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob, e_i, e_n, e_t, e_l = model(textf, acouf, visuf, qmask, umask, lengths)

        # Get probabilities - average over valid sequence elements
        all_prob_flat = all_prob.view(-1, all_prob.size(-1))
        valid_mask = umask.view(-1) > 0
        if valid_mask.sum() > 0:
            probs = torch.softmax(all_prob_flat[valid_mask], dim=-1).mean(dim=0).cpu().numpy()
        else:
            probs = torch.softmax(all_prob_flat, dim=-1).mean(dim=0).cpu().numpy()

        return probs

def main():
    parser = argparse.ArgumentParser(description='Emotion Recognition Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--sample', type=int, default=0, help='Sample index from test set')

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Load test dataset
    test_dataset = IEMOCAPDataset(train=False)

    if args.sample >= len(test_dataset):
        return

    sample_data = test_dataset[args.sample]

    # Predict
    probs = predict_emotion(model, sample_data)

    # Display results
    emotions = ['Angry', 'Excited', 'Frustrated', 'Happy', 'Neutral', 'Sad']

    # Sort by probability
    results = list(zip(emotions, probs))
    results.sort(key=lambda x: x[1], reverse=True)

    for emotion, prob in results:
        print(f"{emotion}: {prob*100:.2f}%")

    print(f"Predicted emotion: {results[0][0]} ({results[0][1]*100:.2f}%)")

if __name__ == '__main__':
    main()