import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. تحميل وتنظيف البيانات
df = pd.read_csv("CBC data_for_meandeley_csv.csv", skiprows=1)

# تنظيف أسماء الأعمدة
df.columns = [col.strip().replace(" ", "_").replace("/", "_") for col in df.columns]

# تحويل القيم الرقمية
for col in df.columns:
    if col != 'Sex':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# حذف الصفوف اللي فيها نواقص
df.dropna(inplace=True)

# إنشاء عمود التصنيف: أنيميا = 1 لو الهيموجلوبين أقل من 12
df['Label'] = df['Hemoglobin'].apply(lambda x: 1 if x < 12 else 0)

# اختيار الأعمدة الرقمية فقط (بدون الأعمدة النصية وLabel)
X = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Label']).values
y = df['Label'].values

# 2. تجهيز البيانات
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=16)

# 3. تعريف موديل PyTorch
class CBCNet(nn.Module):
    def __init__(self, input_dim):
        super(CBCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.output(x)

model = CBCNet(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. تدريب الموديل
for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 5. التقييم
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = torch.argmax(model(xb), dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

accuracy = correct / total
print(f"\n✅ Final Accuracy: {accuracy*100:.2f}%")
