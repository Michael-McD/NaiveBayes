#load "NaiveBayes.fs"
#load "Ingestor.fs"
#load "Helper.fs"

open NaiveBayes.Ingestor
open NaiveBayes.Classifier
open NaiveBayes.Helper
open System.Text.RegularExpressions

let identify (example:SMSType * string) =
    let (docType, content) = example
    match docType with
    |Ham -> printfn "Ham: %s" content
    |Spam -> printfn "Spam: %s" content

let docPath = __SOURCE_DIRECTORY__ + "/Data/SMSSpamCollection"

let dataSet = read docPath
let validationSet, trainingSet  = dataSet.[0..999], dataSet.[1000..]

let matchWords = Regex(@"\w+")
//let matchWordsOnly = Regex(@"[^0-9-!$%^&@#*()_+|~ =`{}\[\]:;\'<>?,.\/""""""]\w+")
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
    
let ham, spam =
    let rawHam, rawSpam =
        trainingSet
        |> Array.partition (fun (lbl, _) -> lbl=SMSType.Ham)
    rawHam |> Array.map snd,
    rawSpam |> Array.map snd
    
// some token feature sets to play with
let spamCount = vocabulary casedTokenizer spam |> Set.count
let hamCount = vocabulary casedTokenizer ham |> Set.count

// filters the validation sets by SMSType 
let spamValidationSet  =
    validationSet
    |> Seq.filter (fun (docType, msg) ->
        docType = SMSType.Spam)

let hamValidationSet  =
    validationSet
    |> Array.filter (fun (docType, _) ->
        docType = SMSType.Ham)

// useful utility to validate one of the training sets
let simpleClassificationTokens = set ["FREE"; "things"]
evaluate smartTokenizer trainingSet simpleClassificationTokens spamValidationSet

// splits the trainingSet in two
let  hamTraining, spamTraining =
    trainingSet
    |> Array.partition (fun (lbl, _) -> lbl=Ham)
// returns a tokenized training set
let tokenized =
    hamTraining
    |> Array.map snd
    |> Array.map casedTokenizer

// test of our countIn function
countIn tokenized "Hi"

// tests our lapace function
laplace 55 3960

// groups of Spam and Ham messages for analysis by the classifier
let groups = learn trainingSet smartTokenizer simpleClassificationTokens

// get all the sms message in from one of the above above groups
let getMap (groups:(SMSType * SMSGroup) []) =
    let (smsType, smsGroup) = groups.[1]
    smsGroup.TokenFrequencies.Count, smsType
    
// use some intuitive words we think will classify Spam.
let keyWordClassifier = train trainingSet smartTokenizer (set ["FREE"; "hurry"; "discount"])
    
// use all the tokens in the training set as the classification feature set
let all = allTokens smartTokenizer trainingSet               // tokenizes all the words in the trainingSet's messages

let allTokenClassifier = train trainingSet smartTokenizer all

// use the top 10% of the training set
let smsStrings = trainingSet |> Array.map snd
let allTokenCount = vocabulary smartTokenizer smsStrings |> Set.count
let top10Percent = smsStrings |> top ((allTokenCount * 15) / 100) casedTokenizer

let top10PercentClassifier = train trainingSet smartTokenizer top10Percent

// rare word tokens
let rarestTokens = smsStrings |> rarest  50 casedTokenizer

let rarestTokensClassifier = train trainingSet smartTokenizer rarestTokens

// top n ham and top spam uncommon tokens
let hamTxts, spamTxts =
    let rawHam, rawSpam =
        trainingSet
        |> Array.partition (fun (lbl, _) -> lbl=SMSType.Ham)
    rawHam |> Array.map snd,
    rawSpam |> Array.map snd

let top10thSpam = spamTxts |> top (spamCount / 10) casedTokenizer
let top10thHam = hamTxts |> top (hamCount / 10) casedTokenizer

let topCommonTokens = Set.intersect top10thSpam top10thHam
let allTopTokens = Set.union top10thSpam top10thHam
let uncommonTokens = Set.difference allTopTokens topCommonTokens

// add the "__PHONE__" and "__TXT__" symbols to the classification token set
let smartTokens =
    uncommonTokens
    |> Set.add "__PHONE__"
    |> Set.add "__TXT__"

let rarestTopTokensClassifier = train trainingSet smartTokenizer smartTokens

// use this to validate any of the above classifiers returned from the train function
validationSet 
|> Seq.filter (fun (smsType,_) -> smsType = Ham) 
|> Seq.averageBy (fun (smsType,sms) -> 
    if smsType = rarestTopTokensClassifier sms 
    then 1.0 
    else 0.0) 
|> printfn "Properly classified Ham: %.5f"

validationSet 
|> Seq.filter (fun (smsType,_) -> smsType = Spam) 
|> Seq.averageBy (fun (smsType,sms) -> 
    if smsType = rarestTopTokensClassifier sms 
    then 1.0 
    else 0.0) 
|> printfn "Properly classified Spam: %.5f"