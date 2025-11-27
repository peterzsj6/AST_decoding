import ast


output_ast=ast.parse(
"""x,y=0,2
for i in range(len(y)):
    x+=1
z=x*y
""")

output=ast.dump(output_ast)
#print(output)
model_ast="""Module(body=[
    Assign(
        targets=[
            Tuple(
                elts=[
                    Name(id='x', ctx=Store()),
                    Name(id='y', ctx=Store())
                ],
                ctx=Store()
            )
        ],
        value=Tuple(
            elts=[
                Num(n=0),
                Num(n=2)
            ],
            ctx=Load()
        )
    ),
    For(
        target=Name(id='i', ctx=Store()),
        iter=Call(
            func=Name(id='range', ctx=Load()),
            args=[
                Call(
                    func=Name(id='len', ctx=Load()),
                    args=[
                        Name(id='y', ctx=Load())
                    ],
                    keywords=[]
                )
            ],
            keywords=[]
        ),
        body=[
            AugAssign(
                target=Name(id='x', ctx=Store()),
                op=Add(),
                value=Num(n=1)
            )
        ],
        orelse=[]
    ),
    Assign(
        targets=[
            Name(id='z', ctx=Store())
        ],
        value=BinOp(
            left=Name(id='x', ctx=Load()),
            op=Mult(),
            right=Name(id='y', ctx=Load())
        )
    )
])

"""

if output >= model_ast:
    ast_length=len(output)
else:
    ast_length=len(model_ast)

matching_count=0
for i in range(ast_length):
    if output[i]==model_ast[i]:
        matching_count+=1

total_matching_percentage=matching_count/ast_length
print("matching is:",total_matching_percentage)

print("Groud_Truth_AST: \n",output)



# import evaluate
# module = evaluate.load("dvitel/codebleu")
# tgt,src=output,model_ast
# res = module.compute(predictions = [tgt], references = [[src]])
# print("The code blue score is:",res)

# from nltk.translate.bleu_score import sentence_bleu

# reference,candidate=output.split(),model_ast.split()
# print('BLEU score -> {}'.format(sentence_bleu(reference, candidate )))
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
X_list = word_tokenize(output) 
Y_list = word_tokenize(model_ast)
X_set = {w for w in X_list } 
Y_set = {w for w in Y_list }

l1 =[];l2 =[]
# form a set containing keywords of both strings 
rvector = X_set.union(Y_set) 
for w in rvector:
    if w in X_set: l1.append(1) # create a vector
    else: l1.append(0)
    if w in Y_set: l2.append(1)
    else: l2.append(0)
c = 0
  
# cosine formula 
for i in range(len(rvector)):
        c+= l1[i]*l2[i]
cosine = c / float((sum(l1)*sum(l2))**0.5)
print("similarity: ", cosine)