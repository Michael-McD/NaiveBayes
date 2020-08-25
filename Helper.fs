namespace NaiveBayes
open NaiveBayes.Ingestor
open NaiveBayes.Classifier

module Helper =

    let docTypePercentages (docTypes:SMSType[]) data =
        let allLen = data |> Array.length
        
        let dtLen dt =
                data
                |> Array.filter( fun (l, _) -> l = dt)
                |> Array.length
                
        docTypes
        |> Array.map( fun dt ->
            let len = dtLen dt
            let proportion = System.Math.Round ((proportion len allLen) * 100.)
            (dt, proportion)
            )

    let vocabulary (tokenizer:Tokenizer) (msg:string seq) =
        msg
        |> Seq.map tokenizer
        |> Set.unionMany
        
    let allTokens (tokenizer:Tokenizer) doc =
        doc
        |> Seq.map snd
        |> vocabulary tokenizer
        
    let evaluate (tokenizer:Tokenizer) trainingData classificationData validationData =
        let classifier = train trainingData tokenizer classificationData
        validationData
        |> Seq.averageBy (fun (docType, msg) ->
            if docType = classifier msg then 1.0 else 0.)
        |> printfn "Correctly classified: %.3f"
        
    let top n (tokenizer:Tokenizer) (docs:string []) =
        let tokenized = docs |> Array.map tokenizer
        let tokens = tokenized |> Set.unionMany 
        tokens
        |> Seq.sortBy (fun t -> - (countIn tokenized t) )
        |> Seq.take n
        |> Set.ofSeq
         
    let rarest n (tokenizer:Tokenizer) (docs:string []) =
        let tokenized = docs |> Array.map tokenizer
        let tokens = tokenized |> Set.unionMany 
        tokens
        |> Seq.sortBy (fun t -> countIn tokenized t)
        |> Seq.take n
        |> Set.ofSeq

            
        
        