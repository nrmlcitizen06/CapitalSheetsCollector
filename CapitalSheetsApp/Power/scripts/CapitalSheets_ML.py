from . import CapitalSheets_Functions
import torch.nn as nn
import joblib

class ImprovedMLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=4, dropout=0.3):
        super(ImprovedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.fc_out = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x  # For simplicity, scale if dims differ
        out = self.fc1(x)
        out = self.relu(self.bn1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(self.bn2(out))
        out = self.dropout(out + residual[:, :out.shape[1]])  # Residual (temporary hack; dims may not align semantically)
        out = self.fc3(out)
        out = self.relu(self.bn3(out))
        out = self.dropout(out)
        return self.fc_out(out)

from sklearn.feature_extraction.text import TfidfVectorizer
# Tokenization Function
def create_tokenizer(texts, max_features=5000, lowercase=True):
    """
    Create a tokenizer with vocabulary mapping.
    
    Args:
        texts (list): Input texts for building vocabulary
        max_features (int): Maximum number of features
        lowercase (bool): Convert to lowercase
    
    Returns:
        dict: Tokenization components
    """
    # Initialize CountVectorizer
    vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=lowercase,
            ngram_range=(1, 3), 
            token_pattern=r'\b\w+\b'
        )
    vectorizer.fit(texts)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Create vocabulary
    vocab = {
        '<pad>': 0,
        '<unk>': 1
    }
    reverse_vocab = {
        0: '<pad>',
        1: '<unk>'
    }
    
    # Add tokens to vocabulary
    for idx, token in enumerate(feature_names, start=2):
        vocab[token] = idx
        reverse_vocab[idx] = token
    
    return {
        'vectorizer': vectorizer,
        'vocab': vocab,
        'reverse_vocab': reverse_vocab
    }


import numpy as np
def add_financial_features(text):
    bal_ind = 1 if any(term in text.lower() for term in ['assets', 'liabilities', 'equity']) else 0
    inc_ind = 1 if any(term in text.lower() for term in ['revenue', 'expenses', 'net income']) else 0
    cash_ind = 1 if any(term in text.lower() for term in ['cash flow', 'operating activities', 'investing']) else 0
    return np.array([bal_ind, inc_ind, cash_ind])  # Append to scaled_features in extract_features



from sklearn.preprocessing import StandardScaler
import torch
# Feature Extraction Function
def extract_features(tokenizer, texts):
    """
    Extract features from texts.
    
    Args:
        tokenizer (dict): Tokenization components
        texts (list): Input texts
    
    Returns:
        tuple: Scaled features and scaler
    """
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Transform texts
    token_matrix = tokenizer['vectorizer'].transform(texts).toarray()

    # Extract financial features for each text
    financial_features = np.array([add_financial_features(text) for text in texts])
    
    # Combine TF-IDF and financial features
    combined_features = np.hstack((token_matrix, financial_features))

    # Scale the combined features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)
    
    return torch.FloatTensor(scaled_features).to(device), scaler

# Prediction Function
def predict_tokens(texts, trained_model_dict):

    # Extract features using saved tokenizer and scaler
    X = trained_model_dict['tokenizer']['vectorizer'].transform(texts).toarray()

    # Add financial features
    financial_features = np.array([add_financial_features(text) for text in texts])
    X_combined = np.hstack((X, financial_features))
    
    X_scaled = trained_model_dict['scaler'].transform(X_combined)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_scaled).to(trained_model_dict['device'])
    
    # Prediction mode
    trained_model_dict['model'].eval()
    with torch.no_grad():
        outputs = trained_model_dict['model'](X_tensor)
        _, predicted = torch.max(outputs, 1)
    
    # Reverse label mapping
    reverse_label_map = {v: k for k, v in trained_model_dict['label_map'].items()}
    return [reverse_label_map[idx] for idx in predicted.cpu().numpy()]


# Neural Network Model
def create_classification_model(input_size):
    """
    Create neural network for classification.
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Size of hidden layers
        num_classes (int): Number of output classes
    
    Returns:
        nn.Module: Neural network model
    """
    return ImprovedMLP(input_size, hidden_size=128, num_classes=4)


# Label Encoding Function
def encode_labels(labels, label_map=None):
    """
    Encode categorical labels.
    
    Args:
        labels (list): Input labels
        label_map (dict, optional): Custom label mapping
    
    Returns:
        tuple: Encoded labels and label mapping
    """
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if label_map is None:
        unique_labels = sorted(set(labels))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    encoded_labels = torch.tensor([label_map[label] for label in labels]).to(device)
    return encoded_labels, label_map


from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_token_classifier(texts, labels, epochs=200, learning_rate=0.001, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = create_tokenizer(texts, max_features=5000, lowercase=True)
    X, scaler = extract_features(tokenizer, texts)
    y, label_map = encode_labels(labels)
    
    X_np = X.cpu().numpy()  # Move to CPU and convert to NumPy
    y_np = y.cpu().numpy()  #Numpy expects CPU not GPU
    
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42, stratify=y_np)
    
    # Convert splits back to tensors and move to device
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    input_size = X_train.shape[1]
    model = ImprovedMLP(input_size, hidden_size=128, num_classes=len(label_map)).to(device)  # Use your ImprovedMLP
    criterion = nn.CrossEntropyLoss()  # Add weights if needed
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = test_correct / test_total
        scheduler.step(test_loss)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch+1}: Train Loss/Acc: {train_loss:.4f}/{train_acc:.4f}, Test Loss/Acc: {test_loss:.4f}/{test_acc:.4f}')
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save best
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    
    # Full evaluation
    print(classification_report(all_labels, all_preds, target_names=list(label_map.keys())))
    print('Confusion Matrix:\n', confusion_matrix(all_labels, all_preds))
    
    return {'model': model, 'tokenizer': tokenizer , 'scaler': scaler, 'label_map': label_map, 'history': history, 'device': device}

import pickle

# Step 2: Train and save the model
def train_and_save_model(model_path="CS_Net", extras_path = "extras.joblib"):
    texts, labels = CapitalSheets_Functions.prepare_training_data()
    print(f"Training on {len(texts)} tables with {len(set(labels))} classes.")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model_dict = train_token_classifier(texts, labels, epochs=200, learning_rate=0.001, batch_size=32)
    
    try:
        torch.save(trained_model_dict['model'].state_dict(), model_path)
        print(f"Model Weights saved to {model_path}")

        extras = {
        'tokenizer': trained_model_dict['tokenizer'],
        'scaler': trained_model_dict['scaler'],
        'label_map': trained_model_dict['label_map'],
        'device': trained_model_dict['device'],
        'history': trained_model_dict['history']
    }
        joblib.dump(extras, extras_path)
        print(f"Joblib extras saved to {extras_path}")

    except Exception as e:
        print(f"Error saving model: {e}")
    
    return trained_model_dict


# Step 3: Load the model
def load_model(model_path="CS_Net", extras_path = "extras.joblib"):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Dynamic device
        input_size = 5003
        model = ImprovedMLP(input_size=input_size, hidden_size=128, num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        extras = joblib.load(extras_path)
        trained_model_dict = {'model': model ,**extras}
        trained_model_dict['device'] = device
        print ("Model Loaded succesfully")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return trained_model_dict


import pandas as pd
    # Step 4: Test the model
def test_model_on_sec_filing(url):
    trained_model = load_model()
    if trained_model is None:
        return
    
    try:
        word_list, processed_data = CapitalSheets_Functions.FindData(url)
        print(f"Extracted {len(processed_data)} tables from SEC filing.")
    except Exception as e:
        print(f"Error fetching/parsing SEC data: {e}")
        return
    
    table_texts = [CapitalSheets_Functions.extract_table_text(df) for df in processed_data if not df.empty]
    print(f"Processed {len(table_texts)} non-empty table texts.")
    
    if not table_texts:
        print("No valid tables to classify.")
        return
    
    predictions = predict_tokens(table_texts, trained_model)
    
    if not predictions:
        print("No predictions made.")
        return
    
    print("\nPrediction Results:")
    for i, (text, pred) in enumerate(zip(table_texts[:5], predictions[:5])):
        #print(f"Table {i+1}: Predicted Category = {pred}")
        #print(f"Sample Text: {text[:100]}...")
        print ("")
    
    try:
        CapitalSheets_Functions.FoundData(predictions)
    except Exception as e:
        print(f"Error in FoundData: {e}")
    
    ABNB = CapitalSheets_Functions.ExtractedData()
    for i, pred in enumerate(predictions):
        if pred == "Balance_Sheet":
            ABNB.balance_sheets.append(processed_data[i].T)
            print (f"Balance_Sheet Index {i}" )
            print ("Balance Sheet:: \n", processed_data[i].T)
        elif pred == "Income_Sheet":
            ABNB.income_sheets.append(processed_data[i].T)
            print (f"Income_Sheet Index {i}")
            print ("Income Sheet:: \n",processed_data[i].T)
        elif pred == "Cashflow_Sheet":
            ABNB.cashflow_sheets.append(processed_data[i].T)
            print (f"Cashflow Sheet Index {i}")
            print ("Cashflow Sheet:: \n",processed_data[i].T)
    
    print(f"\nStored Results:")
    print(f"Balance Sheets: {len(ABNB.balance_sheets)}")
    print(f"Income Sheets: {len(ABNB.income_sheets)}")
    print(f"Cashflow Sheets: {len(ABNB.cashflow_sheets)}")
    
    if ABNB.balance_sheets:
        print("\nSample Balance Sheet:")
        #print(pd.concat(ABNB.balance_sheets, axis=1, ignore_index=True, sort=False).head())
        
    
    # Save predictions for analysis
    results_df = pd.DataFrame({"Text": table_texts, "Predicted": predictions})
    results_df.to_csv("test_predictions.csv", index=False)
    print("Predictions saved to test_predictions.csv")

def CapitalSheets_ML_Called():
    print ("CapitalSheet ML Functions Loaded")

if __name__ == "__main__":
    CapitalSheets_ML_Called()