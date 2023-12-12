documentation_txt = """API_Name = works_list
API_Description = Returns a list of work items matching the request
arguments = [
    {name: applies_to_part, description: Filters for work belonging to any of the provided parts, type: array of strings},
    {name: created_by, description: Filters for work created by any of these users, type: array of strings},
    {name: issue.priority, description: Filters for issues with any of the provided priorities. Allowed values: p0, p1, p2, p3, type: array of strings},
    {name: issue.rev_orgs, description: Filters for issues with any of the provided Rev organizations, type: array of strings},
    {name: limit, description: The maximum number of works to return. The default is '50', type: integer(int32)},
    {name: owned_by, description: Filters for work owned by any of these users, type: array of strings},
    {name: stage.name, description: Filters for records in the provided stage(s) by name, type: array of strings},
    {name: ticket.needs_response, description: Filters for tickets that need a response, type: boolean},
    {name: ticket.rev_org, description: Filters for tickets associated with any of the provided Rev organizations, type: array of strings},
    {name: ticket.severity, description:Filters for tickets with any of the provided severities. Allowed values: blocker, high, low, medium, type: array of strings},
    {name: ticket.source_channel, description: Filters for tickets with any of the provided source channels, type: array of strings},
    {name: type, description: Filters for work of the provided types. Allowed values: issue, ticket, task, type: array of strings},
]

API_Name = summarize_objects
API_Description = Summarizes a list of objects. The logic of how to summarize a particular object type is an internal implementation detail.
arguments = [
    {name: objects, description: List of objects to summarize, type: array of objects},
]

API_Name = prioritize_objects
API_Description = Returns a list of objects sorted by priority. The logic of what constitutes priority for a given object is an internal implementation detail.
arguments = [
    {name: objects, description: A list of objects to be prioritized, type: array of objects},
]

API_Name = add_work_items_to_sprint
API_Description = Adds the given work items to the sprint
arguments = [
    {name: work_ids, description: A list of work item IDs to be added to the sprint., type: array of strings},
    {name: sprint_id, description: The ID of the sprint to which the work items should be added, type: string},
]

API_Name = get_sprint_id
API_Description = Returns the ID of the current sprint
arguments = [
    No arguments required
]

API_Name = get_similar_work_items
API_Description = Returns a list of work items that are similar to the given work item
arguments = [
    {name: work_id, description: The ID of the work item for which you want to ﬁnd similar items, type: string}
]

API_Name = search_object_by_name
API_Description = Given a search string, returns the id of a matching object in the system of record. If multiple matches are found, it returns the one where the conﬁdence is highest.
arguments = [
    {name: query, description: The search string, could be for example customer’s name, part name, user name., type: string}
]

API_Name = create_actionable_tasks_from_text
API_Description = Given a text, extracts actionable insights, and creates tasks for them, which are kind of a work item.
arguments = [
    {name: text, description: The text from which the actionable insights need to be created., type: string}
]

API_Name = who_am_i
API_Description = Returns the ID of the current user
arguments = [
    No arguments required
]
"""

input_query = "List all high severity tickets coming in from slack from customer Cust123 and generate a summary of them."

model_output = """$$PREV[0] = search_object_by_name(query=""Cust123"")
$$PREV[1] = works_list(ticket.rev_org=""$$PREV[0]"", ticket.severity=[""high""], ticket.source_channel=[""slack""])
$$PREV[2] = summarize_objects(objects=""$$PREV[1]"")"""

llm_input = 'You are an intelligent AI agent for generating api calls given a user prompt. To answer a query, one or more api may need to be called. You are provided with api documentation and few examples of queries and their outputs. Read it carefully and understand it, as your output should strictly follow the format of the provided examples and documentation. Also few sub-queries are presented which represent a breakdown of query into a sequence of tasks that you should perform to generate api sequence. The output should be such that if I run the api calls sequentially with those parameters then i will get correct output from the server. Remember to follow the format and only return the sequence of api calls and nothing else.'

examples = '''###Examples:
Query: Summarize issues similar to don:core:dvrv-us-1:devo/0:issue/1
Output: 
$$PREV[0] = get_similar_work_items(work_id=don:core:dvrv-us-1:devo/0:issue/1)
$$PREV[1] = summarize_objects(objects=$$PREV[0])
Query: Find all of my "p3" priority issues and tickets in "support" stage for Rev organization "RevTech", summarize them.
Output: 
$$PREV[0] = whoami()
$$PREV[1] = works_list(owned_by=$$PREV[0], issue.priority=["p2"], applies_to_part=["frontend"])
$$PREV[2] = summarize_objects(objects=$$PREV[1])
Query: Prioritize my P0 issues and add them to the current sprint
Output: 
$$PREV[0] = who_am_i()
$$PREV[1] = works_list(issue.priority=["p0"], owned_by="$$PREV[0]")
$$PREV[2] = prioritize_objects(objects="$$PREV[1]")
$$PREV[3] = get_sprint_id()
$$PREV[4] = add_work_items_to_sprint(work_ids="$$PREV[2]", sprint_id="$$PREV[3]")
Query: Who are the work items which can be used to solve the Solar System?,
Output: Unanswerable'''

feedback_prompt = '''You are an intelligent agent that corrects api sequence code based on feedback. You are provided with documentation, examples and feedback, based on them generate the correct code. Consider this feedback. If the feedback can be integrated into your initial output using the given set of API tools, modify the output to include the feedback and return a new API call sequence. Remember to follow the format specified in the example and return just the api call sequence.'''