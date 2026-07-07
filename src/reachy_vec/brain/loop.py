"""Terminal conversation loop. The voice equivalent lives in oracle.py."""

EXIT_COMMANDS = {"exit", "quit"}


def chat_loop(*, brain, input_fn=input, print_fn=print) -> None:
    print_fn("Reachy chat - ask about your team docs ('exit' to leave).")
    while True:
        try:
            question = input_fn("you> ").strip()
        except EOFError:
            return
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            return
        print_fn(f"reachy> {brain.respond(question)}")
