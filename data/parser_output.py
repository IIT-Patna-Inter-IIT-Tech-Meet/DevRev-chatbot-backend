import json
import itertools
import ast

def json_to_function(output, index = False):
    funclist = ""
    if len(output) == 0:
        return str([])
    for toolind, tool in enumerate(output):
        if index == True:
            line = "$$PREV[{}] = ".format(toolind)
        else:
            line = ""
        if toolind != 0:
            funclist += '\n'
        line = line + tool['tool_name'] + "("
        for ind, arg in enumerate(tool['arguments']):
            if ind != 0:
                line  = line + ", "
            line = line + arg['argument_name'] + "=" 
            if isinstance(arg["argument_value"], str): 
                line += "\""  + arg["argument_value"] + "\""
            else:
                line += str(arg["argument_value"])
        line = line + ")"
        funclist += line
    return funclist


def comma_correction(arglist):
    if len(arglist) == 0:
        return arglist
    # print(arglist)
    correct_list = []
    for i in arglist:
        if "=" in i:
            correct_list.append(i)
        else:
            correct_list[-1] = correct_list[-1] + ", " + i
    return correct_list

def function_to_json(sample, index = True):

    # sample = '$$PREV[0]' + sample.split('$$PREV[0]', 1)[1]
    # print("Sample: \n",sample)
    
    data = {"Output" : ""}
    sample = sample.strip().split('\n')
    outputs = []

    for toolind, tool in enumerate(sample):
        try:
          tooldict = {"tool_name" : "" , "arguments" : []}
          if index == True:
              api_ret_val = "$$PREV[{}] = ".format(toolind)
              tool = tool[len(api_ret_val):]
          tool_name = tool.split('(')[0]
          tool_args = tool[len(tool_name):][1:-1].split(', ')
          tool_args = list(itertools.filterfalse(lambda x: x == '', tool_args))
          tool_args = comma_correction(tool_args)
          arguments = []
          for arg in tool_args:
              if arg.split('=')[1][0] in ['"',"'","["]:
                argument = {"argument_name": arg.split('=')[0], "argument_value": ast.literal_eval(arg.split('=')[1])}
              else:
                argument = {"argument_name": arg.split('=')[0], "argument_value": arg.split('=')[1]}
              arguments.append(argument)
          tooldict["tool_name"] = tool_name
          tooldict["arguments"] = arguments
          outputs.append(tooldict)
        except: 
          pass  

    for i in outputs: 
      while i['tool_name']=="" and i in outputs :
        outputs.remove(i)
      
    data['Output']=outputs

    return data

print(function_to_json('''$$PREV[0] = who_am_i()
$$PREV[1] = works_list(created_by=[\"$$PREV[0]\"],type=[\"issue\",\"ticket\"])
$$PREV[2] = summarize_objects(objects=\"$$PREV[1]\")
$$PREV[3] = create_actionable_tasks_from_text(text=\"$$PREV[2]\")
$$PREV[4] = prioritize_objects(objects=\"$$PREV[3]\")'''))