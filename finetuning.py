from transformers import ( AutoTokenizer,AutoModelForCausalLM,
    Trainer,TrainingArguments,DataCollatorForLanguageModeling
)
from peft import LoraConfig,get_peft_model
import pandas as pd
from datasets import load_dataset

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(
    model_name
)
data = {
    "Question": [
        "What are the best tourist places in India?",
        "Which is the best time to visit Goa?",
        "How can I book a flight online?",
        "What documents are required for international travel?",
        "How do I apply for a tourist visa?",
        "What are the famous places in Hyderabad?",
        "Which hill station is best in South India?",
        "How can I find budget hotels?",
        "What should I pack for a trip?",
        "How can I travel safely?",
        "Which beaches are famous in Goa?",
        "What are the best honeymoon destinations in India?",
        "How do I book train tickets?",
        "Which app is best for hotel booking?",
        "What is travel insurance?",
        "How can I save money while traveling?",
        "What are the top tourist attractions in Delhi?",
        "Which is the cleanest beach in India?",
        "How do I check my flight status?",
        "What are the famous temples in India?",
        "Which places are best for family vacations?",
        "How can I plan a weekend trip?",
        "What are adventure tourism activities?",
        "Which is the best wildlife sanctuary in India?",
        "What is eco-tourism?",
        "How do I exchange currency?",
        "What are the famous waterfalls in India?",
        "Which city is known as the Pink City?",
        "How do I book a cab?",
        "What is the best season to visit Kerala?",
        "Which places are famous in Kashmir?",
        "How can I cancel my hotel booking?",
        "What are the top museums in India?",
        "Which state is famous for backwaters?",
        "How do I find local restaurants?",
        "What is the Golden Triangle tour?",
        "How do I travel on a budget?",
        "Which places are famous in Rajasthan?",
        "How do I check weather before travel?",
        "What are the best trekking destinations?",
        "Which airport is the busiest in India?",
        "How do I renew my passport?",
        "What is a boarding pass?",
        "How early should I reach the airport?",
        "What is check-in luggage?",
        "How can I avoid travel scams?",
        "Which places are UNESCO World Heritage Sites in India?",
        "How do I rent a car?",
        "What are the best food destinations in India?",
        "Which places are best for photography?",
        "How can I find travel deals?",
        "Which is the tallest statue in India?",
        "How do I plan an international trip?",
        "What is the best time to visit Ladakh?",
        "How can I book a cruise?",
        "Which festivals attract tourists in India?",
        "What are the famous forts in Rajasthan?",
        "Which places are best for camping?",
        "How do I travel with pets?",
        "What is solo travel?",
        "How can I find tourist guides?",
        "Which places are famous in Andhra Pradesh?",
        "What are the top places to visit in Telangana?",
        "How do I choose a travel package?",
        "What are the best road trips in India?",
        "How do I get travel updates?",
        "Which is the highest waterfall in India?",
        "How do I use Google Maps offline?",
        "What are the best shopping destinations?",
        "How can I stay healthy while traveling?",
        "Which places are famous for snow?",
        "What are the famous national parks?",
        "How can I avoid jet lag?",
        "Which city is called the City of Lakes?",
        "What is cultural tourism?",
        "How do I book a bus ticket?",
        "Which places are famous for tea gardens?",
        "How do I find public transport?",
        "What are the famous caves in India?",
        "Which place is known for houseboats?",
        "How can I travel sustainably?",
        "Which places are famous for desert tourism?",
        "How do I find emergency contacts while traveling?",
        "What are the best islands in India?",
        "How do I prepare for trekking?",
        "Which places are famous for coffee plantations?",
        "How do I choose travel insurance?",
        "Which city is known as Silicon Valley of India?",
        "How do I find local events?",
        "What are the top beaches in India?",
        "How do I book adventure sports?",
        "Which places are famous for bird watching?",
        "What are the best pilgrimage destinations?",
        "How do I avoid overpacking?",
        "Which is the longest beach in India?",
        "How do I travel during monsoon?",
        "What are the top destinations for students?",
        "How do I manage travel expenses?",
        "What are the best travel apps?",
        "How do I create a travel itinerary?",
        "What are the best places to visit in India during winter?"
    ],

    "Answer": [
        "India offers destinations like Goa, Jaipur, Agra, Kerala, Ladakh, Kashmir, and Ooty.",
        "The best time to visit Goa is from November to February.",
        "Use trusted travel websites or airline apps to book flights.",
        "A passport, visa (if required), tickets, and identification documents are needed.",
        "Apply through the country's official embassy or visa portal.",
        "Charminar, Golconda Fort, Ramoji Film City, and Hussain Sagar are popular attractions.",
        "Ooty, Munnar, Coorg, and Kodaikanal are excellent hill stations.",
        "Compare prices on hotel booking websites and apps.",
        "Pack clothes, medicines, travel documents, chargers, and essentials.",
        "Keep valuables secure and follow local safety guidelines.",
        "Baga Beach, Calangute Beach, and Palolem Beach are famous.",
        "Kerala, Andaman Islands, Goa, and Udaipur are popular honeymoon spots.",
        "Book train tickets using the IRCTC website or app.",
        "Booking.com, MakeMyTrip, Agoda, and Goibibo are popular hotel booking apps.",
        "Travel insurance covers medical emergencies, cancellations, and lost luggage.",
        "Book early, travel during off-season, and use public transportation.",
        "Red Fort, India Gate, Qutub Minar, and Lotus Temple are top attractions.",
        "Radhanagar Beach in the Andaman Islands is one of the cleanest beaches.",
        "Check your airline website or app using your booking reference.",
        "Varanasi, Tirupati, Kedarnath, and Meenakshi Temple are famous temples.",
        "Kerala, Shimla, Ooty, and Mysore are family-friendly destinations.",
        "Choose nearby attractions and prepare a simple itinerary.",
        "Trekking, rafting, camping, and paragliding are popular adventure activities.",
        "Jim Corbett National Park is one of the best wildlife sanctuaries.",
        "Eco-tourism promotes responsible travel while protecting nature.",
        "Exchange currency at banks, airports, or authorized exchange centers.",
        "Jog Falls and Dudhsagar Falls are famous waterfalls.",
        "Jaipur is known as the Pink City.",
        "Use ride-hailing apps like Uber or Ola to book a cab.",
        "September to March is the best time to visit Kerala.",
        "Srinagar, Gulmarg, Pahalgam, and Sonamarg are must-visit places.",
        "Cancel through the hotel booking app or website.",
        "The National Museum and Indian Museum are popular museums.",
        "Kerala is famous for its backwaters.",
        "Use Google Maps or restaurant review apps.",
        "The Golden Triangle includes Delhi, Agra, and Jaipur.",
        "Travel during off-season and compare prices before booking.",
        "Jaipur, Udaipur, Jaisalmer, and Jodhpur are famous destinations.",
        "Use weather forecasting apps before your trip.",
        "Ladakh, Himachal Pradesh, and Uttarakhand offer great trekking.",
        "Indira Gandhi International Airport is India's busiest airport.",
        "Apply online through the Passport Seva Portal.",
        "A boarding pass allows you to enter the aircraft.",
        "Reach the airport at least 2-3 hours before departure.",
        "Check-in luggage is baggage stored in the aircraft cargo hold.",
        "Use trusted transport and avoid sharing personal information.",
        "The Taj Mahal and Hampi are UNESCO World Heritage Sites.",
        "Book rental cars using trusted travel platforms.",
        "Hyderabad, Lucknow, Delhi, and Amritsar are famous food destinations.",
        "Ladakh, Jaipur, and Kerala are excellent photography locations.",
        "Compare travel websites for discounts.",
        "The Statue of Unity is the tallest statue in India.",
        "Plan flights, accommodation, visa, and travel insurance in advance.",
        "June to September is ideal for visiting Ladakh.",
        "Book cruises through travel agencies or cruise operators.",
        "Diwali, Holi, Pushkar Fair, and Durga Puja attract tourists.",
        "Amer Fort, Mehrangarh Fort, and Chittorgarh Fort are famous.",
        "Coorg, Rishikesh, and Spiti Valley are great camping destinations.",
        "Check airline pet travel policies before your trip.",
        "Solo travel means exploring destinations alone.",
        "Hire certified guides through tourism offices.",
        "Araku Valley, Tirupati, and Visakhapatnam are famous places.",
        "Hyderabad, Warangal, and Nagarjuna Sagar are top attractions.",
        "Compare price, itinerary, reviews, and inclusions.",
        "Manali-Leh Highway and East Coast Road are popular road trips.",
        "Follow airline and tourism websites for updates.",
        "Kunchikal Falls is India's highest waterfall.",
        "Download maps for offline use before traveling.",
        "Delhi, Mumbai, Jaipur, and Hyderabad are shopping hubs.",
        "Stay hydrated and carry essential medicines.",
        "Gulmarg, Auli, and Manali are famous snowy destinations.",
        "Kaziranga and Ranthambore are famous national parks.",
        "Adjust your sleep schedule before long flights.",
        "Udaipur is known as the City of Lakes.",
        "Cultural tourism focuses on traditions, festivals, and heritage.",
        "Use government or private bus booking apps.",
        "Munnar and Darjeeling are famous for tea gardens.",
        "Use local transport apps or Google Maps.",
        "Ajanta and Ellora Caves are world-famous.",
        "Alleppey is famous for houseboats.",
        "Reduce plastic use and support local businesses.",
        "Jaisalmer and the Thar Desert are famous for desert tourism.",
        "Save local emergency numbers before your trip.",
        "Andaman and Nicobar Islands and Lakshadweep are popular islands.",
        "Carry proper gear, water, and follow safety instructions.",
        "Coorg and Chikmagalur are famous for coffee plantations.",
        "Choose insurance based on destination and coverage.",
        "Bengaluru is known as the Silicon Valley of India.",
        "Check local tourism websites for events.",
        "Goa, Gokarna, and Andaman have beautiful beaches.",
        "Book through certified adventure tour operators.",
        "Bharatpur Bird Sanctuary is famous for bird watching.",
        "Tirupati, Varanasi, Amritsar, and Rameswaram are major pilgrimage sites.",
        "Pack only essential items and use a checklist.",
        "Marina Beach is India's longest beach.",
        "Carry rain gear and monitor weather forecasts.",
        "Jaipur, Mysore, and Agra are great for educational trips.",
        "Use a travel budget and expense tracking app.",
        "Google Maps, MakeMyTrip, IRCTC, and TripAdvisor are useful travel apps.",
        "List destinations, transport, accommodation, and activities in order.",
        "Kashmir, Auli, Goa, and Rajasthan are excellent winter destinations."
    ]
}

df = pd.DataFrame(data)
df.to_json("travel_tourism_assistant_100_qa.json", orient="records", indent=4)
dataset = load_dataset("json",data_files="travel_tourism_assistant_100_qa.json")
print(dataset)

def formatting(data):
  return {"text":f"Question: {data['Question']} Answer: {data['Answer']}"}
dataset = dataset['train'].map(formatting)
print (dataset)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
def tokenize(prompt):
  result=tokenizer(prompt['text'],padding="max_length",truncation=True,
                   max_length=128)
  result['labels']=result["input_ids"].copy()
  return result
dataset=dataset.map(tokenize)
print(dataset)
config=LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model=get_peft_model(model,config)
print(model.print_trainable_parameters())

data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False)
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="steps",
    fp16=True
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)
trainer.train()
model.save_pretrained("tinyllama_adapter")
tokenizer.save_pretrained("tinyllama_adapter")
device=next(model.parameters()).device
prompt="""
Question:
Which is the best time to visit Goa?
Answer:
"""
inputs=tokenizer(prompt,return_tensors="pt")
inputs={k:v.to(device) for k,v in inputs.items()}
output = model.generate(**inputs,max_new_tokens=100)
print(tokenizer.decode(output[0],skip_special_tokens=True))
