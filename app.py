import gradio as gr
import yfinance as yf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re
import os

print("🚀 Loading model...")

# Your model details
YOUR_LORA_MODEL = "aaysush16/finance-llama-2-7b-lora"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"

# Get HF token from environment (set in Space secrets)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Load tokenizer with token
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with LoRA adapters
try:
    # Configure 4-bit quantization for efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    print("📥 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("📥 Loading your LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, YOUR_LORA_MODEL)
    model.eval()
    model_available = True
    print("✅ Model loaded successfully!")
    
except Exception as e:
    print(f"⚠️ Model loading error: {e}")
    print("Running in fallback mode (stock data only)")
    model_available = False
    model = None

def extract_stock_symbol(text):
    """Extract stock symbol from user message"""
    text_upper = text.upper()
    words = re.findall(r'\b[A-Z]{1,5}\b', text_upper)
    
    # Expanded list of common English words to filter out
    common_words = ['WHAT', 'TELL', 'GIVE', 'SHOW', 'PRICE', 'STOCK', 
                    'THE', 'IS', 'ME', 'ABOUT', 'OF', 'FOR', 'A', 'AN', 'HOW',
                    'THIS', 'THAT', 'THESE', 'THOSE', 'WHEN', 'WHERE', 'WHY',
                    'WHO', 'WHICH', 'CAN', 'COULD', 'WOULD', 'SHOULD', 'WILL',
                    'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID', 'ARE', 'WAS',
                    'WERE', 'BEEN', 'BE', 'BEING', 'IN', 'ON', 'AT', 'TO',
                    'FROM', 'WITH', 'BY', 'AS', 'INTO', 'BUT', 'OR', 'AND',
                    'IF', 'THEN', 'THAN', 'SUCH', 'NO', 'NOT', 'ONLY', 'OWN',
                    'SAME', 'SO', 'SOME', 'FEEL', 'LIKE', 'WANT', 'NEED', 'GET',
                    'MAKE', 'GO', 'KNOW', 'TAKE', 'SEE', 'COME', 'THINK', 'LOOK',
                    'USE', 'FIND', 'WORK', 'MAY', 'MUST', 'SAY', 'HELP']
    
    # Only look for stock symbols if query contains stock-related keywords
    stock_keywords = ['stock', 'share', 'ticker', 'price', 'trading', 'market', 
                      'invest', 'buy', 'sell', 'equity', 'company']
    has_stock_keyword = any(keyword in text.lower() for keyword in stock_keywords)
    
    for word in words:
        if word not in common_words:
            # If has stock keyword OR word looks like known stock pattern
            if has_stock_keyword or len(word) <= 4:
                return word
    
    return None

def get_complete_stock_data(symbol):
    """Fetch ALL relevant stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        if not current_price:
            return None
        
        return {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'current_price': current_price,
            'currency': info.get('currency', 'USD'),
            'previous_close': info.get('previousClose'),
            'open_price': info.get('open'),
            'day_high': info.get('dayHigh'),
            'day_low': info.get('dayLow'),
            'volume': info.get('volume'),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'dividend_yield': info.get('dividendYield'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            'change_percent': info.get('regularMarketChangePercent'),
            'change_amount': info.get('regularMarketChange'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
        }
        
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def format_stock_data(data):
    """Format stock data into readable context for the model"""
    if not data:
        return ""
    
    formatted = f"""LIVE STOCK DATA for {data['company_name']} ({data['symbol']}):
Current Trading Information:
- Current Price: ${data['current_price']:.2f} {data['currency']}
- Previous Close: ${data['previous_close']:.2f}
- Open Price: ${data['open_price']:.2f}
- Day's Range: ${data['day_low']:.2f} - ${data['day_high']:.2f}
- 52 Week Range: ${data['fifty_two_week_low']:.2f} - ${data['fifty_two_week_high']:.2f}
"""
    
    if data['change_amount']:
        formatted += f"- Today's Change: ${data['change_amount']:.2f} ({data['change_percent']:.2f}%)\n"
    
    formatted += f"\nMarket Statistics:\n"
    if data['volume']:
        formatted += f"- Volume: {data['volume']:,}\n"
    if data['market_cap']:
        formatted += f"- Market Cap: ${data['market_cap']:,}\n"
    if data['pe_ratio']:
        formatted += f"- P/E Ratio: {data['pe_ratio']:.2f}\n"
    if data['dividend_yield']:
        formatted += f"- Dividend Yield: {data['dividend_yield']*100:.2f}%\n"
    
    if data['sector']:
        formatted += f"\nCompany Information:\n"
        formatted += f"- Sector: {data['sector']}\n"
        if data['industry']:
            formatted += f"- Industry: {data['industry']}\n"
    
    return formatted

def generate_response(prompt, max_new_tokens=300):
    """Generate response from your fine-tuned model"""
    if not model_available:
        return None
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
        
    except Exception as e:
        print(f"Generation error: {e}")
        return None

def chatbot(message, history):
    """
    Main chatbot function that:
    1. Detects stock symbol in user message
    2. Fetches live data from Yahoo Finance
    3. Sends to your fine-tuned model with context
    4. Returns response
    """
    
    # Extract stock symbol from user message
    stock_symbol = extract_stock_symbol(message)
    
    # If stock symbol detected, fetch live data
    stock_data = None
    if stock_symbol:
        stock_data = get_complete_stock_data(stock_symbol)
    
    # Prepare prompt for your fine-tuned model
    if stock_data:
        # Create rich context with live data
        context = format_stock_data(stock_data)
        full_prompt = f"""{context}
User Question: {message}
Based on the above live stock data, provide a comprehensive and helpful response to the user's question. Use the current market data in your answer."""
        
        # Try to generate with model
        response = generate_response(full_prompt, max_new_tokens=350)
        
        # Fallback if model fails or not available
        if not response:
            response = f"**{stock_data['company_name']} ({stock_data['symbol']})**\n\n"
            response += f"💰 Current Price: **${stock_data['current_price']:.2f}**\n"
            response += f"📊 Change: ${stock_data['change_amount']:.2f} ({stock_data['change_percent']:.2f}%)\n\n"
            response += f"📈 Day's Range: ${stock_data['day_low']:.2f} - ${stock_data['day_high']:.2f}\n"
            response += f"📅 52 Week Range: ${stock_data['fifty_two_week_low']:.2f} - ${stock_data['fifty_two_week_high']:.2f}\n\n"
            if data['market_cap']:
                response += f"💼 Market Cap: ${stock_data['market_cap']:,}\n"
            if stock_data['volume']:
                response += f"📊 Volume: {stock_data['volume']:,}\n"
            if stock_data['pe_ratio']:
                response += f"📈 P/E Ratio: {stock_data['pe_ratio']:.2f}\n"
            if stock_data['sector']:
                response += f"\n🏢 Sector: {stock_data['sector']}\n"
            if stock_data['industry']:
                response += f"🏭 Industry: {stock_data['industry']}\n"
    else:
        # No stock detected - general question
        if model_available:
            response = generate_response(message, max_new_tokens=250)
            if not response:
                response = "I can help you with stock information! Please mention a stock symbol (e.g., AAPL, TSLA, GOOGL) to get live market data."
        else:
            response = "Please ask about a stock by mentioning its symbol.\n\nExamples: 'What is AAPL price?' or 'Tell me about TSLA'"
    
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chatbot,
    title="📊 Finance Chatbot - Live Stock Data",
    description="""
    Ask me about **ANY stock** from major exchanges worldwide! Just mention the stock symbol.
    
    🌍 **Supports thousands of stocks** | 📈 **Real-time data** | 🤖 **AI-powered analysis**
    
    **Examples:**
    - "What is the current price of AAPL?"
    - "Tell me about TSLA stock"
    - "How is NVDA performing?"
    - "Give me info on MSFT"
    """,
    examples=[
        "What is the current price of AAPL?",
        "Tell me about TSLA",
        "How is GOOGL performing today?",
        "Give me complete information on MSFT",
        "What's happening with NVDA stock?",
        "Show me AMZN data"
    ],
    theme=gr.themes.Soft(),
    retry_btn="🔄 Retry",
    undo_btn="↩️ Undo",
    clear_btn="🗑️ Clear",
)

if __name__ == "__main__":
    demo.launch()
