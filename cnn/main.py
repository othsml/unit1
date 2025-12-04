import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

training_data = [
    ("I love this day", 2),
    ("This movie was great", 2),
    ("That is terrible news", 0),
    ("I am so sad today", 0),
    ("The cat sat on the mat", 1),
    ("It is raining outside now", 1),
    ("I feel fantastic and happy", 2),
    ("What a horrible experience", 0),
    ("This tastes like nothing", 1),
    ("Amazing job, truly superb", 2),
    ("I hate spiders", 0)
]

all_words = []
for sentence, label in training_data:
    all_words.extend(sentence.lower().split())

word_to_ix = {"<unk>": 0}
for word in set(all_words):
    if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_CLASSES = 3

def sentence_to_indices(sentence):
    indices = [word_to_ix.get(word, word_to_ix["<unk>"])
               for word in sentence.lower().split()]
    return torch.tensor(indices, dtype=torch.long)

class SimpleSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(SimpleSentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_sentence_indices):
        embeds = self.embedding(input_sentence_indices)
        sentence_vector = torch.mean(embeds, dim=0)
        sentiment_scores = self.linear(sentence_vector)
        return sentiment_scores

EMBEDDING_DIM = 10
model = SimpleSentimentModel(VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

NUM_EPOCHS = 100

print("Starting model training (This is the AI learning)...")

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for sentence, label in training_data:
        indices = sentence_to_indices(sentence)
        target = torch.tensor([label], dtype=torch.long)
        model.zero_grad()
        sentiment_scores = model(indices)
        loss = loss_function(sentiment_scores.unsqueeze(0), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1:3d}, Average Loss: {total_loss / len(training_data):.4f}')

print("Training complete! The model is ready to predict.")
print("-" * 40)

def predict_sentiment(input_sentence):
    model.eval()
    with torch.no_grad():
        indices = sentence_to_indices(input_sentence)
        scores = model(indices)
        predicted_index = torch.argmax(scores).item()

    if predicted_index == 0:
        return "NEGATIVE"
    elif predicted_index == 1:
        return "NEUTRAL"
    else:
        return "POSITIVE"

if __name__ == "__main__":
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    print(f"Model Vocabulary Size: {VOCAB_SIZE} words")
    print(f"The model is trained to recognize 3 classes: {sentiment_map}")
    print("\n--- Try It Out! ---")


    user_input = input("Enter a sentence (or type 'quit' to exit): ")
    prediction = predict_sentiment(user_input)
    print(f"Sentiment Prediction: ===> {prediction} <===\n")