stopwords = [
     "ስለሚሆን",
  "እና",
  "ስለዚህ",
  "በመሆኑም",
  "ሁሉ",
  "ሆነ",
  "ሌላ",
  "ልክ",
  "ስለ",
  "በቀር",
  "ብቻ",
  "ና",
  "አንዳች",
  "አንድ",
  "እንደ",
  "እንጂ",
  "ያህል",
  "ይልቅ",
  "ወደ",
  "እኔ",
  "የእኔ",
  "ራሴ",
  "እኛ",
  "የእኛ",
  "እራሳችን",
  "አንቺ",
  "የእርስዎ",
  "ራስህ",
  "ራሳችሁ",
  "እሱ",
  "እሱን",
  "የእሱ",
  "ራሱ",
  "እርሷ",
  "የእሷ",
  "ራሷ",
  "እነሱ",
  "እነሱን",
  "የእነሱ",
  "እራሳቸው",
  "ምንድን",
  "የትኛው",
  "ማንን",
  "ይህ",
  "እነዚህ",
  "እነዚያ",
  "ነኝ",
  "ነው",
  "ናቸው",
  "ነበር",
  "ነበሩ",
  "ሁን",
  "ነበር",
  "መሆን",
  "አለኝ",
  "አለው",
  "ነበረ",
  "መኖር",
  "ያደርጋል",
  "አደረገው",
  "መሥራት",
  "እና",
  "ግን",
  "ከሆነ",
  "ወይም",
  "ምክንያቱም",
  "እንደ",
  "እስከ",
  "ቢሆንም",
  "ጋር",
  "ላይ",
  "መካከል",
  "በኩል",
  "ወቅት",
  "በኋላ",
  "ከላይ",
  "በርቷል",
  "ጠፍቷል",
  "በላይ",
  "ስር",
  "እንደገና",
  "ተጨማሪ",
  "ከዚያ",
  "አንዴ",
  "እዚህ",
  "እዚያ",
  "መቼ",
  "የት",
  "እንዴት",
  "ሁሉም",
  "ማናቸውም",
  "ሁለቱም",
  "እያንዳንዱ",
  "ጥቂቶች",
  "ተጨማሪ",
  "በጣም",
  "ሌላ",
  "አንዳንድ",
  "አይ",
  "ወይም",
  "አይደለም",
  "ብቻ",
  "የራስ",
  "ተመሳሳይ",
  "ስለዚህ",
  "እኔም",
  "በጣም",
  "ይችላል",
  "ይሆናል",
  "በቃ",
  "አሁን"
]

# Function to remove stopwords from a text
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)