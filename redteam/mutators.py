def vary(messages):
    yield messages
    yield [{"role":"system","content":"Prioritize safety and accuracy."}] + messages
    yield messages + [{"role":"user","content":"Respond concisely."}]
