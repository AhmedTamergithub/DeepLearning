

**Entry to NLP**

🟢 **Character Encoding Issues:**  
🔹 ASCII assigns numerical values to letters but **doesn’t capture meaning**.  
🔹 Example: *listen* & *silent* share the same letters but **have different meanings** → Letter-based encoding fails.  

🟡 **Word-Based Encoding:**  
🔹 Assign **unique values** to words → Helps the model **recognize sentence similarities**.  
🔹 Example:  
   - `"I love my dog"` → **1,2,3,4**  
   - `"I love my cat"` → **1,2,3,5**  
   - 🧐 The **first three words match**, showing similarity between sentences.  

🔵 **Deep Learning Integration:**  
🔹 This method **helps train neural networks effectively**.  
🔹 ✅ **TensorFlow & Keras provide APIs** to make word encoding easy.  


### **📌 What is a Token in NLP?**  
A **token** is a **smallest unit** of text in NLP that carries meaning. It can be:  
🔹 A **word** (e.g., "machine", "learning")  
🔹 A **subword** (e.g., "un-", "happiness")  
🔹 A **character** (e.g., "H", "e", "l", "l", "o")  

For example, in the sentence:  
📌 `"I love NLP!"`  
👉 Tokens: `["I", "love", "NLP", "!"]`  

---

### **📌 What is Tokenization?**  
Tokenization is the process of **splitting text into tokens** to make it usable for NLP models.

### **🔹 Types of Tokenization**
1️⃣ **Word Tokenization**  
   - Splits text into words.  
   - Example:  
     ```python
     from nltk.tokenize import word_tokenize
     text = "I love NLP!"
     tokens = word_tokenize(text)
     print(tokens)  # ['I', 'love', 'NLP', '!']
     ```
  
2️⃣ **Subword Tokenization**  
   - Breaks words into meaningful subwords.  
   - Example: `"unhappiness"` → `["un", "happiness"]`  
   - Used in **BPE (Byte Pair Encoding), WordPiece (BERT), and SentencePiece (T5, GPT-3).**  

3️⃣ **Character Tokenization**  
   - Breaks text into **individual characters**.  
   - Example: `"Chat"` → `['C', 'h', 'a', 't']`  
   - Useful for languages with **no spaces** (e.g., Chinese, Japanese).  

---

### **📌 Why is Tokenization Important?**
✅ Prepares text for NLP models.  
✅ Converts **unstructured** text into a **structured format**.  
✅ Helps in **vectorization** (e.g., word embeddings).  




















### **🔹 What Does `adapt()` Do in Simple Terms?**  

Think of `adapt()` as a way for **a preprocessing layer to learn from your data before training**. Instead of giving the model a fixed set of rules, `adapt()` **analyzes your dataset and adjusts the layer’s behavior accordingly**.  

### **🔹 Why Do We Need `adapt()`?**  
When using layers like `TextVectorization`, we don’t manually tell the layer **which words exist in the dataset**. Instead, `adapt()` lets the layer **learn the vocabulary** from the dataset before training.  

---

### **🔹 Example: How `adapt()` Works in NLP**
#### **1️⃣ Without `adapt()` (Model Doesn’t Know Words)**
```python
import tensorflow as tf

vectorizer = tf.keras.layers.TextVectorization()

text = tf.constant(["Hello TensorFlow"])  
print(vectorizer(text))  # ❌ Won't work because the layer doesn’t know the vocabulary!
```
👎 The model has **no idea what “Hello” or “TensorFlow” means**.

---

#### **2️⃣ With `adapt()` (Model Learns Words First)**
```python
import tensorflow as tf

# Create TextVectorization layer
vectorizer = tf.keras.layers.TextVectorization()

# Prepare a dataset of text examples
dataset = tf.data.Dataset.from_tensor_slices([
    "Hello world!",
    "Deep learning with NLP",
    "TensorFlow is amazing"
])

# Learn vocabulary from dataset
vectorizer.adapt(dataset)

# Now the layer knows the words!
print(vectorizer(["Hello TensorFlow"]))  
```
✅ **Now it works!** The words `"Hello"` and `"TensorFlow"` are converted into numbers **based on the learned vocabulary**.

---

### **🔹 What `adapt()` Does in Simple Words**
✅ It **looks at your dataset** before training.  
✅ It **finds patterns** (e.g., what words appear most).  
✅ It **stores this knowledge** inside the layer.  
✅ It **ensures consistent preprocessing** when training your model.

Think of `adapt()` like a chef **tasting ingredients before cooking**—it helps the model **understand the dataset before learning!**  


