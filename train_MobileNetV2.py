
from datetime import datetime
from model import MobileNetV2, get_difference_in_seconds, append_row_to_csv

DEFAULT_DATE_TIME_FORMAT = "%Y%m%d-%H%M%S"

complete_run_time_details_file_name = "MobileNetV2_complete_run_timing_" + datetime.now().strftime(DEFAULT_DATE_TIME_FORMAT) + ".csv"
complete_run_timing_file = "./trainingTiming/" + complete_run_time_details_file_name


def main():
    """
    Script entrypoint
    """
    t_start = datetime.now()
    header = ["Start Time", "End Time", "Duration (s)"]
    row = [t_start.strftime(DEFAULT_DATE_TIME_FORMAT)]

    dnn = MobileNetV2()

    # show class indices
    print('****************')
    for cls, idx in dnn.train_batches.class_indices.items():
        print('Class #{} = {}'.format(idx, cls))
    print('****************')

    print(dnn.model.summary())

    dnn.train(t_start, epochs=dnn.num_epochs, batch_size=dnn.batch_size, training=dnn.train_batches,validation=dnn.valid_batches)

    # save trained weights
    dnn.model.save(dnn.file_weights + 'old')

    dnn.model.save_weights(dnn.file_weights)
    with open(dnn.file_architecture, 'w') as f:
        f.write(dnn.model.to_json())

    t_end = datetime.now()
    difference_in_seconds = get_difference_in_seconds(t_start, t_end)

    row.append(t_end.strftime(DEFAULT_DATE_TIME_FORMAT))
    row.append(str(difference_in_seconds))

    append_row_to_csv(complete_run_timing_file, header)
    append_row_to_csv(complete_run_timing_file, row)


if __name__ == "__main__":
    main()