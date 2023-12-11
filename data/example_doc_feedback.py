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
