class ResponseParser:
    """
    Parses LLM responses to extract a single function call using a rigid textual format.

    The LLM must output exactly one function call at the end of its response.
    Do NOT use JSON or XML. Use rfind to locate the final markers.
    """

    BEGIN_CALL = "----BEGIN_FUNCTION_CALL----"
    END_CALL = "----END_FUNCTION_CALL----"
    ARG_SEP = "----ARG----"
    VALUE_SEP = "----VALUE----"

    # Students should include this exact template in the system prompt so the LLM follows it.
    response_format = f"""
your_thoughts_here
...
{BEGIN_CALL}
function_name
{ARG_SEP}
arg1_name
{VALUE_SEP}
arg1_value (can be multiline)
{ARG_SEP}
arg2_name
{VALUE_SEP}
arg2_value (can be multiline)
...
{END_CALL}

DO NOT CHANGE ANY TEST! AS THEY WILL BE USED FOR EVALUATION.
"""

    def parse(self, text: str) -> dict:
        """
        Parse the function call from `text` using string.rfind to avoid confusion with
        earlier delimiter-like content in the reasoning.

        Returns a dictionary: {"thought": str, "name": str, "arguments": dict}
        
        TODO(student): Implement this function using rfind to parse the function call
        """
        begin_idx = text.rfind(self.BEGIN_CALL)
        end_idx = text.rfind(self.END_CALL)
        
        if begin_idx == -1 or end_idx == -1 or begin_idx >= end_idx:
            raise ValueError(f"Could not find function call markers in text. BEGIN_CALL at {begin_idx}, END_CALL at {end_idx}")
        
        thought = text[:begin_idx].strip()
        
        call_section = text[begin_idx + len(self.BEGIN_CALL):end_idx].strip()
        
        parts = call_section.split(self.ARG_SEP)
        if not parts:
            raise ValueError("Function call section is empty")
        
        function_name = parts[0].strip()
        
        arguments = {}
        for part in parts[1:]:
            if self.VALUE_SEP not in part:
                continue
            
            arg_name, arg_value = part.split(self.VALUE_SEP, 1)
            arg_name = arg_name.strip()
            arg_value = arg_value.strip()
            arguments[arg_name] = arg_value
        
        return {
            "thought": thought,
            "name": function_name,
            "arguments": arguments
        }
