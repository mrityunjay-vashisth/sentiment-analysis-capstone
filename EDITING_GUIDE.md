# How to Edit and Run the Notebooks

This guide provides multiple ways to edit and run the Jupyter notebooks in this project.

---

## 🚀 Quick Start Options

### Option 1: Edit in Google Colab (Easiest - No Setup)

**Best for:** Quick edits, trying things out, sharing with others

1. **Open in Colab:**
   - Click the "Open in Colab" badge in the README, or
   - Go to https://colab.research.google.com/
   - Click "GitHub" tab
   - Enter: `mrityunjay-vashisth/sentiment-analysis-capstone`
   - Select the notebook you want to edit

2. **Run the notebook:**
   - All cells will run in Google's cloud environment
   - Click "Runtime" → "Run all" to execute all cells
   - Or use `Shift+Enter` to run cells one by one

3. **Edit the notebook:**
   - Click any cell to edit it
   - Add new cells with `+ Code` or `+ Text` buttons
   - Make your changes

4. **Save your changes:**
   - **To your Google Drive:** File → Save a copy in Drive
   - **To GitHub (if you have write access):** File → Save a copy in GitHub
   - **Download to your computer:** File → Download → Download .ipynb

**Direct Colab Links:**
- EDA Notebook: https://colab.research.google.com/github/mrityunjay-vashisth/sentiment-analysis-capstone/blob/main/notebooks/01_Exploratory_Data_Analysis.ipynb
- Results Notebook: https://colab.research.google.com/github/mrityunjay-vashisth/sentiment-analysis-capstone/blob/main/notebooks/02_Model_Results_Analysis.ipynb

**Note:** Colab won't have access to the data files by default. You'll need to either:
- Upload the CSV files from `data/splits/` to Colab
- Mount your Google Drive and place files there
- Modify the file paths in the notebook

---

### Option 2: Edit Locally with Jupyter (Full Control)

**Best for:** Full control, working with actual data, serious development

#### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/mrityunjay-vashisth/sentiment-analysis-capstone.git
cd sentiment-analysis-capstone
```

#### Step 2: Set Up Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Install Jupyter if not already included
pip install jupyter notebook
```

#### Step 3: Launch Jupyter

```bash
# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab for a more modern interface
jupyter lab
```

This will open a browser window with the file explorer. Navigate to `notebooks/` and click on any notebook to open it.

#### Step 4: Edit and Run

- Click on any cell to edit
- Run cells with `Shift+Enter`
- Add new cells with `+` button or `Insert` menu
- Save with `Ctrl+S` (or `Cmd+S` on Mac)

#### Step 5: Save Your Changes

```bash
# Stage your changes
git add notebooks/

# Commit your changes
git commit -m "Updated analysis in EDA notebook"

# Push to your fork (if you forked the repo)
git push origin main
```

---

### Option 3: Edit with VS Code (Modern IDE Experience)

**Best for:** Developers familiar with VS Code, integrated development

#### Step 1: Install VS Code and Extensions

1. Download and install [VS Code](https://code.visualstudio.com/)
2. Install the **Jupyter extension** from the Extensions marketplace
3. Install the **Python extension**

#### Step 2: Open the Project

```bash
# Clone and open in VS Code
git clone https://github.com/mrityunjay-vashisth/sentiment-analysis-capstone.git
cd sentiment-analysis-capstone
code .
```

#### Step 3: Set Up Python Environment

1. Open the Command Palette (`Cmd+Shift+P` or `Ctrl+Shift+P`)
2. Type "Python: Create Environment"
3. Select "Venv"
4. Choose your Python interpreter
5. Check the box to install dependencies from `requirements.txt`

#### Step 4: Open and Edit Notebooks

1. Navigate to `notebooks/` folder
2. Click on any `.ipynb` file
3. VS Code will render it as an interactive notebook
4. Select your Python environment (bottom right)
5. Click "Run All" or run cells individually

**Advantages:**
- Excellent code completion and IntelliSense
- Integrated terminal and Git
- Side-by-side editing
- Variable explorer and debugging

---

### Option 4: Edit on GitHub Directly (Minor Changes)

**Best for:** Quick fixes to markdown cells, fixing typos

1. Go to https://github.com/mrityunjay-vashisth/sentiment-analysis-capstone
2. Navigate to `notebooks/` folder
3. Click on the notebook file
4. Click the pencil icon (✏️) to edit
5. Make your changes in the JSON editor
6. Scroll down and commit changes

**Warning:** GitHub's editor shows the raw JSON format, which can be confusing. Only use this for simple text changes.

---

## 📊 Working with the Data

The notebooks expect data files in specific locations:

```
data/splits/
├── sentiments_clean_train.csv
├── sentiments_clean_test.csv
├── clothing_reviews_train.csv
└── clothing_reviews_test.csv
```

### If You Don't Have the Data Files:

**Option A: Use Sample Data**
Create smaller CSV files with the same structure:

```python
import pandas as pd

# Create sample data
sample_data = pd.DataFrame({
    'id': ['twitter_1', 'twitter_2', 'twitter_3'],
    'text': ['Great product!', 'Not happy with this', 'Its okay I guess'],
    'created_at': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'topic': ['product', 'product', 'product'],
    'label': ['positive', 'negative', 'neutral']
})

# Save to the expected location
sample_data.to_csv('data/splits/sentiments_clean_train.csv', index=False)
```

**Option B: Modify the Notebook**
Change the file paths in the notebook to point to your data:

```python
# Original
social_train = pd.read_csv('../data/splits/sentiments_clean_train.csv')

# Modified
social_train = pd.read_csv('/path/to/your/data.csv')
```

---

## 🔄 Syncing Your Changes

### If You Forked the Repository:

```bash
# Add upstream remote (original repo)
git remote add upstream https://github.com/mrityunjay-vashisth/sentiment-analysis-capstone.git

# Fetch latest changes
git fetch upstream

# Merge upstream changes
git merge upstream/main

# Push to your fork
git push origin main
```

### Creating a Pull Request:

If you made improvements and want to contribute back:

1. Fork the repository on GitHub
2. Make your changes in your fork
3. Push to your fork
4. Go to the original repository
5. Click "New Pull Request"
6. Select your fork and branch
7. Describe your changes and submit

---

## 💡 Tips for Editing

### Best Practices:

1. **Run cells in order** - Notebooks depend on previous cells being executed
2. **Restart kernel regularly** - `Kernel → Restart & Clear Output` to start fresh
3. **Save often** - `Ctrl+S` / `Cmd+S` to save your work
4. **Use markdown cells** - Document your changes with markdown cells
5. **Comment your code** - Help others (and future you) understand your changes

### Common Issues:

**"Module not found" errors:**
```bash
# Install missing package
pip install package-name

# Or add to requirements.txt and reinstall
pip install -r requirements.txt
```

**"File not found" errors:**
- Check your current working directory
- Verify file paths are correct (relative to notebook location)
- Use `!pwd` in a cell to see current directory
- Use `!ls ../data/splits/` to list available files

**Kernel crashes:**
- Restart the kernel: `Kernel → Restart`
- Clear all outputs: `Cell → All Output → Clear`
- Try running cells one at a time

**Out of memory:**
- Close other notebooks
- Restart the kernel
- Work with smaller data samples
- Use `del variable_name` to free memory

---

## 🎓 Learning Resources

**Jupyter Notebooks:**
- [Jupyter Notebook Basics](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html)
- [Markdown Cheat Sheet](https://www.markdownguide.org/cheat-sheet/)

**Python Libraries Used:**
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

**Git and GitHub:**
- [GitHub Docs](https://docs.github.com/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)

---

## 🆘 Getting Help

If you run into issues:

1. **Check the error message** - Read it carefully, it often tells you what's wrong
2. **Google the error** - Copy the error message and search
3. **Check existing issues** - Look at [GitHub Issues](https://github.com/mrityunjay-vashisth/sentiment-analysis-capstone/issues)
4. **Ask for help** - Create a new issue with:
   - What you were trying to do
   - What happened instead
   - Error messages (full text)
   - Your environment (OS, Python version)

---

## 📝 Example Workflow

Here's a complete example of editing a notebook:

```bash
# 1. Clone the repo
git clone https://github.com/mrityunjay-vashisth/sentiment-analysis-capstone.git
cd sentiment-analysis-capstone

# 2. Create environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook

# 4. Open notebooks/01_Exploratory_Data_Analysis_executed.ipynb
# 5. Make your edits in the browser
# 6. Save the notebook (Ctrl+S)
# 7. Close Jupyter (Ctrl+C in terminal)

# 8. Commit your changes
git add notebooks/01_Exploratory_Data_Analysis_executed.ipynb
git commit -m "Added additional analysis section"
git push origin main
```

---

Happy editing! 🎉
