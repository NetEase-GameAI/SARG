CANARD: A Dataset for Question-in-Context Rewriting


Ahmed Elgohary and Denis Peskov and Jordan Boyd-Graber
"Can You Unpack That? Learning to Rewrite Questions-in-Context"
In Empirical Methods in Natural Language Processing, 2019.


train.json
dev.json
test.json

    Training, development and testing splits of CANARD. Each json file is an array of question, 
    context, and rewrite objects. Each object has the following fields: 
        History: an array of previous dialog utterances in the same order they appear in the
                 dialog. The first two utterances are always the Wikipedia article title followed
                by the section title.

        Question: the target question to be rewritten.

        Rewrite: reference rewrite.

        QuAC_dialog_id: the id of QuAC dialog used to generate the example.

        Question_no: the number of the question as in appears in the full dialog (the fist
        question has question_no1) .



multiple_refs.json
    A 100 rewrite pairs each consists of two rewrites of the same question provided by two
    different crowd workers.


CANARD is distributed under the CC BY-SA 4.0 license.
For more details see http://canard.qanta.org or contact Ahmed Elgohary (elgohary@cs.umd.edu)