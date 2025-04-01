package hk.ust.csit5970;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

/**
 * First-pass Mapper: Counts word frequencies.
 * For each word in the input, it emits (word, count) where count is the number of occurrences
 * of the word in that particular line.
 */
public class CORMapper1 extends Mapper<LongWritable, Text, Text, IntWritable> {
    private static final Text WORD = new Text();
    private static final IntWritable COUNT = new IntWritable();

    @Override
    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        HashMap<String, Integer> wordCount = new HashMap<String, Integer>();
        // Clean the document and tokenize it
        String cleanDoc = value.toString().replaceAll("[^a-z A-Z]", " ");
        StringTokenizer tokenizer = new StringTokenizer(cleanDoc);
        
        // Count occurrences of each word in this line
        while (tokenizer.hasMoreTokens()) {
            String word = tokenizer.nextToken().toLowerCase();
            if (wordCount.containsKey(word)) {
                wordCount.put(word, wordCount.get(word) + 1);
            } else {
                wordCount.put(word, 1);
            }
        }
        
        // Emit each word with its count
        for (Map.Entry<String, Integer> entry : wordCount.entrySet()) {
            WORD.set(entry.getKey());
            COUNT.set(entry.getValue());
            context.write(WORD, COUNT);
        }
    }
} 