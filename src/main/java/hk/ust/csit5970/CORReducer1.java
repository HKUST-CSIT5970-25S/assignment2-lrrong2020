package hk.ust.csit5970;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * First-pass Reducer: Sums up frequencies for each word.
 * For each word, it sums up all counts and emits (word, totalCount).
 */
public class CORReducer1 extends Reducer<Text, IntWritable, Text, IntWritable> {
    private static final IntWritable SUM = new IntWritable();

    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
        int sum = 0;
        // Sum up all counts for the current word
        for (IntWritable value : values) {
            sum += value.get();
        }
        SUM.set(sum);
        context.write(key, SUM);
    }
} 