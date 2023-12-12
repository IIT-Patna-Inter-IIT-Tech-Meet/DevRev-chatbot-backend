import re
import json

class ToolArgument():
    def __init__(self, ArgumentName, ArgumentDescription, ArgumentType, ArgumentValueExamples = []):
        self.ArgumentName = ArgumentName
        self.ArgumentDescription = ArgumentDescription
        self.ArgumentType = ArgumentType
        self.ArgumentValueExamples = ArgumentValueExamples
    
    def get_json(self):
        return {"Argument Name" : self.ArgumentName,  "Argument Description" : self.ArgumentDescription , "Argument Type" : self.ArgumentType , "Argument Value Examples" : self.ArgumentValueExamples}
    
class Tool():
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.arguments = []
        
    def add_argument(self, ArgumentName, ArgumentDescription, ArgumentType, ArgumentValueExamples = []):
        self.arguments.append(ToolArgument(ArgumentName, ArgumentDescription, ArgumentType, ArgumentValueExamples).get_json())
    
    def get_json(self):
        tool_json = {"API Name" : self.name , "API Description":  self.description, "API arguments": self.arguments}
        return tool_json
    
class ToolDatabase():
    def __init__(self):
        self.ToolList = []
        
    def read_existing_json(self, path_to_exisiting_json):
        with open(path_to_exisiting_json, 'r') as f: # read file
            self.ToolList = json.load(f)['ToolList']       
        
    def save_to_new_json_file(self, path_to_new_json):
        with open(path_to_new_json, 'w') as f: # write file
            tool_dict = {'ToolList' : self.ToolList}
            f.write(self.get_json())
        
    def add_tool(self, tool: Tool):
        self.ToolList.append(tool)
        
    def get_json(self):
        return json.dumps(self, default= lambda o: o.__dict__, indent= 4)
    
    def function_names(self):
        display = [a["API Name"] for a in self.ToolList]
        return display
    
    def retrive_documentation_from_name(self, names):
        documentation = [0 for i in range(len(names))]
        for ind, name in enumerate(names):
            for doc in self.ToolList:
                if doc.name == name:
                    documentation[ind] = self.get_documentation_as_string(doc.get_json())
                    break
        return documentation
    
    def get_documentation_as_string(self, doc):
        tool = ""
        for key, value in doc.items():
            if key != "API arguments":
                tool += str(key) + " : " + str(value) + "\n"
            else:
                tool += str(key) + " :\n"
                for arg in value:
                    tool += "{ "
                    for argkey , argvalue in arg.items():
                        tool += "\"" + str(argkey) + "\" : " + str(argvalue) + ", "
                    tool += "}, \n"
        return tool
    
    def remove_tool(self, name):
        new_ToolList = []
        for tool in self.ToolList:
            if tool["API Name"] != name :
                new_ToolList.append(tool)
        self.ToolList = new_ToolList
        return self.ToolList
        

#init this class with all_examples.json and pass name when u need to delete correspondiing examples and use get_all_examples to get the list of examples
class Examples():
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.example_list = []
        
        with open(file_name, 'r') as f: # read file
            self.example_list = json.load(f)
            
    def for_tool_remove_examples(self , name):
        new_example_list = []
        for example in self.example_list:
            if re.search(" = {}\(".format(name), example["Output"]) == None :
                new_example_list.append(example)
        self.example_list = new_example_list
        return self.example_list
    
    def get_all_examples(self):
        return self.example_list
    
    def save_current_examples_in_json(self, name):
        with open(name, 'w') as f: # write file
            json.dump(self.example_list, f)