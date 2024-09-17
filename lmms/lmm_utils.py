
prompt_template_open = """\
Here are two videos of an action with the following description: "{action}".
{video_representation_description}

Return a list of 'differences' in how the action is being performed. 
Each difference should have a 'description' that is a specific statements that more true in one video compared to the other video. 
Then there is a 'pred' which is 'a' if the statement applies more to video 'a' and 'b' if it applies more to video 'b'.
The differences should be visual and about how the action is performed. 

Return a json like this:
{ 
    "differences" : [
        {
            "description" : ....,
            "pred" : "a|b",
        }, ...
    ]
}
"""


video_rep_description_2_videos = """We have passed 'video a' and 'video b' to the prompt."""
video_rep_description_1_video = """We have passed in 1 video into the prompt, which is the concatenation of 'video a' and then 'video b'."""
video_rep_description_2_grids = None # """We have passed in 1 video into the prompt, which is the concatenation of 'video a' and then 'video b'."""
video_rep_description_2_sequences = None # """We have passed in 1 video into the prompt, which is the concatenation of 'video a' and then 'video b'."""


prompt_template_closed = """\
"""