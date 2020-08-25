namespace NaiveBayes

module Classifier =
    
    type Token = string
    type Tokenizer = string -> Token Set // fun string = token[]
    type TokenizedDoc = Token Set

    // proportion of the group members within the whole, a map of a word (aka token) and the number of occurrences in the group
    type SMSGroup = {Proportion:float; TokenFrequencies:Map<Token, float>} 

    // given a precomputed group of token frequency's return the tokens value if present else 0.0
    let tokenScore (group:SMSGroup) (token:Token) =
        if group.TokenFrequencies.ContainsKey token
        then log group.TokenFrequencies.[token]
        else 0.0

    let score (document:TokenizedDoc) (group:SMSGroup) =
        let scoreDoc = tokenScore group // partial application
        group.Proportion + (document |> Seq.sumBy scoreDoc)  // use a Seq to iterate a Set

    let classify (groups:(_*SMSGroup)[]) (tokenizer:Tokenizer) (text:string) =  // using _ we can pass in any discriminated union type as a label
        let tokenized = tokenizer text
        groups
        |> Array.maxBy (fun (label, group) -> score tokenized group)
        |> fst

    let proportion count total = float count / float total
    let laplace count total = float (count + 1) / float (total + 1)

    let countIn (group:TokenizedDoc seq) (token:Token) =
        group
        |> Seq.filter (Set.contains token)
        |> Seq.length
    
    let analyze (group:TokenizedDoc seq) (totalDocs:int) (classificationTokens:Token Set) =
        let groupSize = group |> Seq.length

        let score token = 
            let count = countIn group token
            laplace count groupSize
        
        let scoredTokens =
            classificationTokens
            |> Set.map (fun token -> token, score token)
            |> Map.ofSeq

        let groupProportion = proportion groupSize totalDocs
        {
            Proportion = groupProportion         
            TokenFrequencies = scoredTokens
        }

    // learn from the training set/document
    let learn (docs:(_ * string)[]) (tokenizer:Tokenizer) (classificationTokens:Token Set) =
        let length = docs.Length
        docs
        |> Array.map(fun (label, msg) -> label, tokenizer msg)
        |> Seq.groupBy fst
        |> Seq.map(fun (label, group) -> label, group |> Seq.map snd)
        |> Seq.map(fun (label, group) -> label, analyze group length classificationTokens)
        |> Seq.toArray
        
    let train (sms_txts:(_ * string)[]) (tokenizer:Tokenizer) (classificationTokens:Token Set) =
        let groups = learn sms_txts tokenizer classificationTokens
        let classifier = classify groups tokenizer
        classifier
    
