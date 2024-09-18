## prompt templates for all eval modes
prompt_template_open = """\
Here are two videos of an action with the following description: "{action_description}".
{video_representation_description}

Return a list of 'differences' in how the action is being performed. 
Each difference should have a 'description' that is a specific statements that more true in one video compared to the other video. 
The differences should be visual and about how the action is performed.
Suggest no more than {n_differences} differences.

Then there is a 'prediction' which is 'a' if the statement applies more to video a and 'b' if it applies more to video b.
The differences should be visual and about how the action is performed. 

Return a json like this:
{ 
    "differences" : [
        {
            "description" : ....,
            "prediction" : "a|b",
        }, ...
    ]
}
"""

prompt_template_closed = """\
"""

## prompt templates for explaining how the video is represented
video_rep_description_2_videos = """\
We have passed 'video a' and 'video b' as video files to the prompt."""
video_rep_description_1_video = """\
We have passed in 1 video into the prompt, which is the concatenation of 'video a' and then 'video b'."""
video_rep_description_2_grids = None  # """We have passed in 1 video into the prompt, which is the concatenation of 'video a' and then 'video b'."""
video_rep_description_frames = """\
We have passed a sequence of images into the prompt. 
The first {vid0_nframes} are video A. The last {vid1_nframes} are video B. 
The frame rate is the same and is {fps}.
"""
video_rep_description_2_grids = """\
We have passed two images into the prompt. 
The first image is a grid of images showing the frames of video A row-wise. 
The first image is a grid of images showing the frames of video B row-wise. 
The frame rate is the same.
"""
