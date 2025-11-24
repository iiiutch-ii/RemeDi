def get_statistics(data):
    """
    data is a"""
    mask_id=126336
    trajectory_inputs = data["trajectory_inputs"]
    remask_in_each_step = []
    for in1, in2 in zip(trajectory_inputs[:-1], trajectory_inputs[1:]):
        # find where in1 != mask_id and in2 == mask_id
        sum_remask_here = (in1 != mask_id) & (in2 == mask_id)
        # sum the number of True values in sum_remask_here
        sum_remask_here = sum_remask_here.sum().item()
        remask_in_each_step.append(sum_remask_here)
        
    return {
        "sum_remasked_tokens": sum(remask_in_each_step),
        "remask_in_each_step": remask_in_each_step,
    }


def get_average_statistics(statistics_list):
    """
    statistics_list is a list of statistics dictionaries
    """
    num_statistics = len(statistics_list)
    return {
        "sum_remasked_tokens": sum(stat["sum_remasked_tokens"] for stat in statistics_list) / num_statistics,
        "remask_in_each_step": [
            sum(stat["remask_in_each_step"][i] for stat in statistics_list) / num_statistics
            for i in range(len(statistics_list[0]["remask_in_each_step"]))
        ],
    }
