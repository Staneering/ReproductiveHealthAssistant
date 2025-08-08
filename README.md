# Chainlit Chatbot Project

This project is a Chainlit-powered chatbot that leverages LLMs and custom knowledge sources. Follow these instructions to set up and run the project locally or share it with others.

## 🚀 Features

- Customizable chatbot logic (e.g., drug analyzer, first aid, music, or sexual/reproductive health)
- Loads knowledge from web URLs, PDFs, and more
- Built with [Chainlit](https://docs.chainlit.io) for rapid prototyping and sharing

## 🛠️ Setup Instructions

### 1. **Clone the Repository**
```sh
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. **Create a Virtual Environment**
```sh
python -m venv venv
```
Activate it:
- On Windows:
  ```
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  source venv/bin/activate
  ```

### 3. **Install Requirements**
```sh
pip install -r requirements.txt
```

### 4. **Set Up Environment Variables**
Some LLM providers (e.g., Groq) require API keys.  
Set your API key as an environment variable (example for Windows):
```sh
set GROQ_API_KEY=your_groq_api_key_here
```
Or edit the code to insert your key directly (not recommended for production).

### 5. **Add Your Data Sources**
- Place your PDFs, Excel files, or other data in the project directory as needed.
- Edit the source lists in the code (e.g., `DRUG_SOURCES` in `chatboturl.py`) to point to your files or URLs.

### 6. **Run the Chainlit App**
```sh
chainlit run chatboturl.py
```
- The app will be available at [http://localhost:8000](http://localhost:8000).

### 7. **(Optional) Make the App Public with ngrok**
- [Download ngrok](https://ngrok.com/download) and run:
  ```sh
  ngrok http 8000
  ```
- Share the generated public URL.

## 📝 Customizing the Welcome Screen

Edit the `chainlit.md` file at the root of the project to change the welcome message shown to users.

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[MIT](LICENSE) (or your chosen license)

---

**Happy coding**