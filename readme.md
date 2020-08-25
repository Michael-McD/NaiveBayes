# Naive Bayes Classifier

This [Naive Bayes classifer](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) is based upon Mathias Branderwinder's "ML Projects for .NET Developers" spam classifier and is implemented in F#.

The program is split into three main modules:
1) __Injestor.fs__ data ingestor - reading in and formatting training data
2) __NaiveBayes.fs__ Naive Bayes algorithm - the implementation
3) __Script.fsx__ data shaping scripts - extracting salient features within the raw data

With the __Helper.fs__ module 
* applying a `tokenizer` to an SMS message and returning a set of tokenized words. 
* defining a helper `evalulate` method to bring all the necessary pieces together to evaluate the classifier with any feature set chosen for training.

## The Ingestor

The ingestor reads labeled spam/ham message data (from UCL's [ML Repo](http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)) and transforms it into a format suitable for processing by the classifier.

```f#
let type SMSType = | Ham | Spam             // the type of SMS message i.e. it's label in the source data
let type SMS = string                       // the message itself

let type LabeledMessage = (SMSType * SMS)   // an F# Tuple describing a labelled SMS message 
```
A labelled message consists of an SMS Type and a string stored in an F# Tuple . This format will support the process of feature extraction refinement.

 ## The Naive Bayes theorem:
  
   `P (A | B) = ( P (B | A) * P (A) ) / P (B)` 
   
Probability P of A given that B, is equal to the probability of B given A times the probability 
of A, which is then divided by the probability of B.

In our case then (for example) using the word FREE as a feature that determines a text being spam: 
P ( SMS is Spam | SMS contains "FREE") = (P (SMS contains "FREE" | SMS is Spam) * P (SMS is Spam) ) / P (SMS contains "FREE")

The algorithm therefore needs to determine how often a feature ("FREE" in this case) occurs in spam SMS's within the world of SMS (or our training dataset from UCL's ML repo mentioned above). 
Then multiply this by the probability of SMS messages being spam, and divide this by SMS's containing "FREE".

**Laplace Smoothing**: is used to mitigate against a feature (a word in our case e.g. "FREE") occurring in one group within our dataset
over determining the classification. So we use Lapalce Smoothing to prevent 
rare words from being assigned assigning 100% confidence. In our case, to an SMS message being in the Spam group.  
The result being that rare words will have a low probability of nearly but not quite zero.

Here's how it works: 

`P (Spam group contains "FREE") = (1 + count of Spam SMS's containing "FREE") / (1 + count of Spam SMS's)`

and in F# (where count = count of Spam containing feature, and total = count of Spam SMS's)
   
```f#
    let laplace count total = float (count + 1) / float (total + 1)
```
**Using Logarithms**: as our formula involves multiplying probability values below 1 we're likely
to encounter rounding issues. So instead we transform the computations into  a
SUM (addition) instead of a Product (multiplication) by using logarithms.

```f#
    let tokenScore (group:DocGroup) (token:Token) =
        if group.TokenFrequencies.ContainsKey token
        then log group.TokenFrequencies.[token]
        else 0.0

    let score (document:TokenizedDoc) (group:DocGroup) =
        let scoreDoc = tokenScore group                      // partial application
        group.Proportion + (document |> Seq.sumBy scoreDoc)  // use a Seq to iterate a Set
```

Here you can see the `tokenScore` returns the log of a tokens occurrences `log group.TokenFrequencies.[token]` and in the `score`
function scores are summed `Seq.sumBy scoreDoc`.

## Our Naive Bayes Algorithm 

The types we will use are important to consider:

```f#
    type Token = string
    type Tokenizer = string -> Token Set
    type SMSGroup = {Proportion:float; TokenFrequencies:Map<Token, float>} 
```
`Token` is obviously a string  
`Tokenizer` is a function signature, takes a string and returns a set of Tokens e.g. "this is a string" -> {"this";"is";"a";"string"}  
`SMSGroup` is the proportion of SMS messages in a group (Spam/Ham), along with a Map (Dictionary) of the each feature or token (i.e. a word) and a value indicating the number of time the feature appears in the group.

Our implementation starts out with the `let train (trainingDoc:(SMSType * string)[]) (tokenizer:Tokenizer) (classificationTokens:Token Set) `  
The `train` functions returns a classifier which can then be used to classify an SMS text message as Spam or Ham.  To build the classifier `train`
takes a labelled collection of Ham and Spam messages, the training set, a `tokenizer`, to turn SMS messages into a set of tokens (words), and a feature set used to select between Spam and Ham.

Two other functions worth mentions are `learn` and `analyze`.  `learn` really breaks the training set into groups and call's `analyze`
on each group. `analyze` builds the internal model used by the classification algorithm.

`analyze` returns an array of `SMSGroup` records = {Proportion:float; TokenFrequencies:Map<Token, float>}.  Each record has the following return signature:

```f#
{Proportion:float; TokenFrequencies:Map<Token, float>}
```

This is exactly the data we need to apply the Naive Bayes algorithm: the number of SMS texts in a group (the Proportion), 
a Map (dictionary) consisting of a Token and a count of it's occurrences in the group. 
So after training we have the following model:

```f#
let clissificatinFeatures = set ["5";"A";"Do";"Hi";"I"]
learn trainingSet casedTokenizer clissificatinFeatures
val it : (SMSType * SMSGroup) [] =
  [|(Ham,
     { Proportion = 0.8657630083
       TokenFrequencies =
                         map
                           [("5", 0.01035092148); ("A", 0.01514768998);
                            ("Do", 0.02044938147); ("Hi", 0.01413784398);
                            ("I", 0.3049734915)] });
    (Spam,
     { Proportion = 0.1342369917
       TokenFrequencies =
                         map
                           [("5", 0.02276422764); ("A", 0.02276422764);
                            ("Do", 0.02113821138); ("Hi", 0.02276422764);
                            ("I", 0.05365853659)] })|]
```
Using the model the classifier can classify any SMS using the combined
probabilities of the classification features (i.e. words) occurring in the SMS text message.  
E.g. "Do" will have a probability of 0.02113821138 occurring in Spam as opposed to a probability of 0.02044938147 of occurring in a Ham SMS text message.
Thus will be add to the combined probability of the message being classified as Spam.


### *Setting things up before we start experimenting with the various classification features we can derive from the data.*

__Note: this is where most of the time working with this classifier will be spent. I.e. extracting features from the data to 
use as classification features.__


Extracting some validation SMS messages.  We use the 1st 1000 message from the raw data and the rest (4000 +) for training.

```f#
let dataSet = read docPath
let validationSet, trainingSet  = dataSet.[0..999], dataSet.[1000..]
```

Tokenization: we experiment with three tokenizers, un-cassed, cassed, and one that transforms phone numbers and SMS service numbers into labels.
```f#
let matchWords = Regex(@"\w+")
let uncasedTokenizer (text:string) =
    text
    |> matchWords.Matches
    |> Seq.cast<Match>
    |> Seq.map(fun m -> m.Value)
    |> Set.ofSeq
    
let casedTokenizer (text:string) =
    text
    |> matchWords.Matches
    |> Seq.cast<Match>
    |> Seq.map(fun m -> m.Value)
    |> Set.ofSeq
    
// functions to transform phone and txt numbers into common labels
let phoneWords = Regex(@"0[7-9]\d{9}")
let phone (word:string) =
    match (phoneWords.IsMatch word) with
    | true -> "__PHONE__"
    | false -> word
    
let txtWords = Regex(@"\b\d{5}\b")
let txt (word:string) =
    match (txtWords.IsMatch word) with
    | true -> "__TXT__"
    | false -> word
    
// binds the above transformations to the tokenizer
let smartTokenizer = casedTokenizer >> Set.map phone >> Set.map txt
    
```

Evaluation: to evaluate the classifier function returned from the training function
```f#
let classifier = train sms_txts cassedTokenizer classificationTokens
```
This function returns a classifier which is a function which takes a string and classifies it like so
```f#
classifier "FREE money today!!!"
val it : SMSType = Spam
```

To evaluate each message in the entire validation set and get a count of the SMS messages correctly classified 
(according to the messages label) we use the following in the Scripts.fsx script file (changing the filter's SMSType).  

```f#
validationSet 
|> Seq.filter (fun (smsType,_) -> smsType = SMSType.Ham) 
|> Seq.averageBy (fun (smsType,sms) -> 
    if smsType = classifier sms 
    then 1.0 
    else 0.0) 
|> printfn "Properly classified Ham: %.5f"
```
## Classification Features
This is where we will be using the data shaping capabilities of F# to analyse the data so that we can extract 
and test various classification feature sets.

We can try guessing at some key words that could be used as successful classification features.
```f#
let keyWordClassifier = train sms_txts cassedTokenizer (set ["FREE"; "hurry"; "discount"]
```
The validation set can now be run against this `classifier`  for both Spam and Ham messages.  The Spam validation code is repeated for clarity:
```f#
validationSet 
|> Seq.filter (fun (smsType,_) -> smsType = SMSType.Spam) 
|> Seq.averageBy (fun (smsType,sms) -> 
    if smsType = keyWordClassifier sms 
    then 1.0 
    else 0.0) 
|> printfn "Properly classified Spam: %.5f"
```
Here's the result:
* Properly classified Ham: 1.00000    
* Properly classified Spam: 0.15789 

Hardly any Spam detected, looks like almost everything is being classified as Ham (including Spam).

Lets try throwing in every word.  We'll use the `allTokens` function (in Helper.fs) to get them.
```f#
let all = allTokens smartTokenizer trainingSet
let allTokenClassifier = train trainingSet smartTokenizer all
```
* Properly classified Ham: 0.77712  
* Properly classified Spam: 0.98684  

Lots of Spam detected but not very good on non Spam i.e. this is incorrectly classifying a lot of Ham messages as Spam which is not what we want.

Trying the top 10 percent of tokens (or words) in the training set:
```f#
let smsStrings = trainingSet |> Array.map snd                                   // get snd elements of training tuple (SMSType * smsTxt)  array i.e. the text
let allTokenCount = vocabulary smartTokenizer smsStrings |> Set.count           // vocabulary returns a union of all the word tokens which is then counted
let top10Percent = smsStrings |> top ((allTokenCount * 15) / 100) casedTokenizer// top orders the smsStrings and takes the top n (15% in this case)
let top10PercentClassifier = train trainingSet smartTokenizer top10Percent
```
* Properly classified Ham: 0.87854  
* Properly classified Spam: 0.98026  

Still miss classifying Ham a bit too much.  Rather have fewer spam messages get through than classify a good ham message as spam and thus potentially lose an important message.

Rare words:
```f#
let rarestTokens = smsStrings |> rarest  50 smartTokenizer                  // rarest does what it say... see the Helper.fs
let rarestTokensClassifier = train trainingSet smartTokenizer rarestTokens
```
* Properly classified Ham: 1.00000
* Properly classified Spam: 0.06579  

Every thing pretty much is now classified as ham... not good.

Lets try merging the top _n_ Ham and the top _n_ Spam tokens so we have a even distribution on ham and spam specific words.
```f#
let spamTxts, hamTxts =                                                // partitionHam and Spam SMS txt messages
    let rawSpam, rawHam = trainingSet |> Array.partition (fun (lbl, _) -> lbl=Ham)
    rawHam |> Array.map snd, rawSpam |> Array.map snd

let rarestSpam1 = spamTxts |> rarest  50 smartTokenizer                // get the top 50 from SpamTxt and HamTxt
let rarestHam1 = hamTxts |> rarest 50 smartTokenizer
let rarestTopTokens = Set.union rarestSpam1 rarestHam1                 // and merge the unique tokens

let rarestTopTokensClassifier = train trainingSet smartTokenizer rarestTopTokens
```
Validating the `rarestTopTokensClassifier` classifier yields:  
* Properly classified Ham: 0.99764  
* Properly classified Spam: 0.23684  

Better, but not finding many Spam messages.

Lets try the top non intersecting tokens.
```f#
let topSpam = spamTxts |> top (spamCount / 10) casedTokenizer
let topHam = hamTxts |> top (hamCount / 10) casedTokenizer

let topCommonTokens = Set.intersect topSpam1 topHam1
let allTopTokens = Set.union topSpam1 topHam1     
let uncommonTokens = Set.difference allTopTokens topCommonTokens      // the allTokens minus the common ones

let rarestTopTokensClassifier = train trainingSet smartTokenizer uncommonTokens
```
Yields:
* Properly classified Ham: 0.96580
* Properly classified Spam: 0.94079

Much better.  

There is one final improvement we can make.  Note that the training `train` function is using the `smartTokenizer`, which
if you remember, is replacing phone and txt numbers with "_PHONE_" & "_TXT_" in the trainingSet.  However these symbols aren't
in our classification set yet so lets add them and see what difference this makes.
```f#
let smartTokens =
    uncommonTokens
    |> Set.add "__PHONE__"
    |> Set.add "__TXT__"

let rarestTopTokensClassifier = train trainingSet smartTokenizer smartTokens
```
Gives us a slightly better classification of Spam, with an acceptable detection of Ham messages.
* Properly classified Ham: 0.96580
* Properly classified Spam: 0.95395


## Conclusion

The Naive Bayes algorithm is one of the simplest probabilistic classifiers - it's implementation has taken us less than
70 lines of F# code to implement. However the selection of so called feature/predictors has proven to be more more interesting, 
requiring a detailed analysis of the data space. Often with suprising results, for example adding more data to the classification 
feature set made the predictions worse or that key terms we intuitively thought might do that trick were amongst the worst.

This is where F# has proved to be an excellent choice allowing us to extract potential classification feature from a variety of patterns within the data.

A final thought being that given the classification feature detections required a lot of analysis of patterns within the data set an idea might be to combine such a classifier with a hidden feature detection neural network.
