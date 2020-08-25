namespace NaiveBayes

open System.IO

module Ingestor =

    type SMSType =
        | Spam
        | Ham

    let matchLabel label =
        match label with
        | "ham" -> Ham
        | "spam" -> Spam
        | _ -> failwith "Unknown label."

    let parseLine (line: string) =
        let split = line.Split('\t')
        let label = split.[0] |> matchLabel
        let message = split.[1]
        (label, message)

    let read dataFilePath =
        File.ReadAllLines dataFilePath
        |> Array.map (fun l -> parseLine l)
