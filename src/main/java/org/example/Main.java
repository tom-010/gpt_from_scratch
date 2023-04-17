package org.example;
import java.io.*;
import java.nio.file.*;
import java.util.*;

import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.*;
import org.json.simple.parser.ParseException;


/*

GPT
Generative Pretrained Transformer


https://github.com/openai/tiktoken

*/

class Encoder {

    Map<Integer, Integer> codePointToEncoding = new HashMap<>();
    Map<Integer, Integer> encodingToCodePoint = new HashMap<>();

    List<Integer> alphabet;

    void init(String text) {
        var codePoints = Arrays.stream(text.codePoints().toArray()).boxed().toList();
        var unique = new HashSet<>(codePoints);
        alphabet = unique.stream().sorted().toList();

        for(int i=0; i<alphabet.size(); i++) {
            codePointToEncoding.put(alphabet.get(i), i);
            encodingToCodePoint.put(i, alphabet.get(i));
        }
    }

    List<Integer> encode(String s) {
        return s.codePoints().map(c -> codePointToEncoding.get(c)).boxed().toList();
    }

    String decode(List<Integer> encoded) {
        return encoded.stream()
                .map(e -> encodingToCodePoint.get(e))
                .filter(Objects::nonNull)
                .collect( StringBuilder::new , StringBuilder::appendCodePoint , StringBuilder::append)
                .toString();
    }

    String getAlphabetString() {
        return alphabet.stream()
                .filter(Objects::nonNull)
                .collect( StringBuilder::new , StringBuilder::appendCodePoint , StringBuilder::append)
                .toString();
    }

    String decode(Integer integer) {
        return decode(List.of(integer));
    }

}


class BigramLanguageModel {

    private Random ran = new Random();

    Map<Integer, Counter2> table = new HashMap<>();


    void addToTable(int input, int expected) {
        if(!table.containsKey(input)) {
            table.put(input, new Counter2());
        }
        table.get(input).add(expected);
    }

    void fit(Sample sample) {
        for(int i=0; i<sample.size(); i++) {
            addToTable(sample.input.get(i), sample.expected.get(i));
        }
    }

    int nextToken(List<Integer> context) {
        int current = context.get(context.size()-1);
        return table.get(current).draw();

    }

    private int randomToken() {
        var options = table.keySet().stream().toList();
        var idx = ran.nextInt(options.size());

        return options.get(idx);
    }

    public void showTable(Encoder encoder) {
        var keys = table.keySet().stream().sorted().toList();
        for(int key : keys) {
            System.out.println("=====");
            System.out.println(encoder.decode(List.of(key)));
            table.get(key).showTally(encoder);
        }
    }

}


class DataSource3 {

    Encoder encoder;
    String path;

    List<Integer> encoded;

    List<Integer> trainingData;

    List<Integer> testData;

    Random ran = new Random();



    DataSource3(String path) {
        this.path = path;
    }

    DataSource3() {
        this.path = null;
    }

    void read() throws IOException {
        String content = readFile();
        encoder = new Encoder();
        encoder.init(content);
        encoded = encoder.encode(content);
        splitData();
    }

    private String readFile() throws IOException {
        File file = new File(path);
        InputStream inputStream = new FileInputStream(file);

        StringBuilder resultStringBuilder = new StringBuilder();
        try (BufferedReader br
                     = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = br.readLine()) != null) {
                resultStringBuilder.append(line).append("\n");
            }
        }
        return resultStringBuilder.toString();
    }

    void splitData() {
        int i = (int) (0.9 * encoded.size());
        trainingData = encoded.subList(0, i);
        testData = encoded.subList(i, encoded.size());
    }

    Sample getTrainingSample(int contextLength) {
        return drawFromList(trainingData, contextLength);
    }

    Sample getTestSample(int contextLength) {
        return drawFromList(trainingData, contextLength);
    }

    private Sample drawFromList(List<Integer> list, int contextLength) {
        int max = list.size() - (contextLength+1);
        int position = ran.nextInt(max);

        return new Sample(
                list.subList(position, position + contextLength),
                list.subList(position+1, position + contextLength + 1));
    }



}


class RandomGenerator {

    List<Integer> alphabet;
    Random random = new Random();

    RandomGenerator(List<Integer> alphabet) {
        this.alphabet = alphabet;
    }

    Integer nextToken() {

        var index = random.nextInt(alphabet.size());
        return alphabet.get(index);
    }


}


public class Main {

    public static void main(String[] args) throws IOException {

        DataSource3 ds = new DataSource3("input.txt");
        ds.read();

        var encoder = ds.encoder;

        var s = "Hii there";

        BigramLanguageModel m = new BigramLanguageModel();

        for(int i=0; i<1000; i++) {
            var d = ds.getTrainingSample(8);
            m.fit(d);
        }


        for(int i=0; i<150; i++) {
            var context = List.of(0);
            System.out.print(encoder.decode(m.nextToken(context)));
        }

    }
















        public static void main3(String[] args) throws MalformedModelException, IOException, ParseException, TranslateException {
        Path modelDir = Paths.get("./");
        Model model = Model.newInstance("gpt_gpu");
        model.load(modelDir);
        Predictor<List<Integer>, Integer> predictor = model.newPredictor(new GptTranslator());
        predictor.predict(List.of(0));
    }


    public static void main2(String[] args) throws IOException, MalformedModelException, TranslateException {
        DataSource ds = new DataSource("input.txt");
        ds.read();
        var encoder = ds.encoder;

        Path modelDir = Paths.get("./");
        Model model = Model.newInstance("gpt");
        model.load(modelDir);

        Predictor<List<Integer>, Integer> predictor = model.newPredictor(new GptTranslator());

        int contextSize = 64;
        List<Integer> context = new LinkedList<Integer>();
        context.add(0); // initial token

        for(int i=0; i<500; i++) {
            int nextToken = predictor.predict(List.of(50));
            context.add(nextToken);
            while(context.size() > contextSize) {
                context.remove(0);
            }
            //System.out.print(encoder.decode(nextToken));
        }
        System.out.println();
    }


}












// You are all resolved rather to die than to famish?

// [1 2 3 5 4 3 5 7 8]
// [3 5 4 3]
// [5 4 3 5]

class Sample {
    List<Integer> input;
    List<Integer> expected;

    Sample(List<Integer> input, List<Integer> expected) {
        this.input = input;
        this.expected = expected;
    }

    int size() {
        return input.size();
    }

    @Override
    public String toString() {
        return "Sample{" +
                "input=" + input +
                ", expected=" + expected +
                '}';
    }
}

class DataSource {

    Encoder encoder;
    String path;

    List<Integer> encoded;

    List<Integer> trainingData;

    List<Integer> testData;

    Random ran = new Random();



    DataSource(String path) {
        this.path = path;
    }

    DataSource() {
        this.path = null;
    }

    void read() throws IOException {
        String content = readFile();
        encoder = new Encoder();
        encoder.init(content);
        encoded = encoder.encode(content);
        splitData();
    }

    private String readFile() throws IOException {
        File file = new File(path);
        InputStream inputStream = new FileInputStream(file);


        StringBuilder resultStringBuilder = new StringBuilder();
        try (BufferedReader br
                     = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = br.readLine()) != null) {
                resultStringBuilder.append(line).append("\n");
            }
        }
        return resultStringBuilder.toString();
    }

    void splitData() {
        int i = (int) (0.9 * encoded.size());
        trainingData = encoded.subList(0, i);
        testData = encoded.subList(i, encoded.size());
    }

    Sample getTrainingSample(int contextLength) {
        return drawFromList(trainingData, contextLength);
    }

    Sample getTestSample(int contextLength) {
        return drawFromList(trainingData, contextLength);
    }

    private Sample drawFromList(List<Integer> list, int contextLength) {
        int max = list.size() - (contextLength+1);
        int position = ran.nextInt(max);

        return new Sample(
                list.subList(position, position + contextLength),
                list.subList(position+1, position + contextLength + 1));
    }



}

class GptTranslator implements Translator<List<Integer>, Integer> {

    @Override
    public NDList processInput(TranslatorContext ctx, List<Integer> input) {
        NDManager manager = ctx.getNDManager();
        int[] primitive = input.stream().mapToInt(Integer::intValue).toArray();
        NDArray array = manager.create(primitive);
        return new NDList (array);
    }

    @Override
    public Integer processOutput(TranslatorContext ctx, NDList logits) {
        var probabilities = logits.get(0).softmax(-1).toFloatArray();

        return ctx.getNDManager().randomMultinomial(1, logits.get(0).softmax(-1)).getInt();

        /*var m = new Multinomial<Integer>();
        for(int i=0; i<probabilities.length; i++) {
            m.add(i, probabilities[i]);
        }
        return m.sample();*/
    }
}

class BashTranslator {

}





class Sample2 {
    List<Integer> input;
    List<Integer> expected;

    Sample2(List<Integer> input, List<Integer> expected) {
        this.input = input;
        this.expected = expected;
    }

    int size() {
        return input.size();
    }

    @Override
    public String toString() {
        return "Sample{" +
                "input=" + input +
                ", expected=" + expected +
                '}';
    }
}

class DataSource2 {

    Encoder encoder;
    String path;

    List<Integer> encoded;

    List<Integer> trainingData;

    List<Integer> testData;

    Random ran = new Random();


    DataSource2(String path) {
        this.path = path;
    }

    void read() throws IOException {
        String content = readFile();
        encoder = new Encoder();
        encoder.init(content);
        encoded = encoder.encode(content);
        splitData();
    }

    private String readFile() throws IOException {
        File file = new File(path);
        InputStream inputStream = new FileInputStream(file);


        StringBuilder resultStringBuilder = new StringBuilder();
        try (BufferedReader br
                     = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = br.readLine()) != null) {
                resultStringBuilder.append(line).append("\n");
            }
        }
        return resultStringBuilder.toString();
    }

    void splitData() {
        int i = (int) (0.9 * encoded.size());
        trainingData = encoded.subList(0, i);
        testData = encoded.subList(i, encoded.size());
    }

    Sample getTrainingSample(int contextLength) {
        return drawFromList(trainingData, contextLength);
    }

    Sample getTestSample(int contextLength) {
        return drawFromList(trainingData, contextLength);
    }

    private Sample drawFromList(List<Integer> list, int contextLength) {
        int max = list.size() - (contextLength+1);
        int position = ran.nextInt(max);

        return new Sample(
                list.subList(position, position + contextLength),
                list.subList(position+1, position + contextLength + 1));
    }
}

class Counter2 {

    Map<Integer, Integer> tally = new HashMap<>();
    List<Integer> elements = new LinkedList<>();

    private Random ran = new Random();

    void add(Integer i) {
        if(!tally.containsKey(i)) {
            tally.put(i, 0);
        }
        tally.put(i, tally.get(i) + 1);
        elements.add(i);
    }

    Integer draw() {
        return elements.get(ran.nextInt(elements.size()));
    }

    public void showTally(Encoder encoder) {
        var keys = tally.keySet().stream().sorted().toList();
        for(int key : keys) {
            System.out.print(encoder.decode(List.of(key)));
            System.out.print("=");
            System.out.print(tally.get(key) + "; ");

        }
        System.out.println();

        System.out.println(encoder.decode(elements));
    }


}

class BigramLanguageModel2 {

    private Random ran = new Random();

    Map<Integer, Counter2> table = new HashMap<>();

    void addToTable(int input, int expected) {
        if(!table.containsKey(input)) {
            table.put(input, new Counter2());
        }
        table.get(input).add(expected);
    }

    void fit(Sample sample) {
        for(int i=0; i<sample.size(); i++) {
            addToTable(sample.input.get(i), sample.expected.get(i));
        }
    }

    int nextToken(List<Integer> context) {
        int current = context.get(context.size()-1);
        return table.get(current).draw();

    }

    private int randomToken() {
        var options = table.keySet().stream().toList();
        var idx = ran.nextInt(options.size());

        return options.get(idx);
    }

    public void showTable(Encoder encoder) {
        var keys = table.keySet().stream().sorted().toList();
        for(int key : keys) {
            System.out.println("=====");
            System.out.println(encoder.decode(List.of(key)));
            table.get(key).showTally(encoder);
        }
    }
}


class NGramLanguageModel2 {

    private Random ran = new Random();
    int n;

    NGramLanguageModel2(int n) {
        this.n = n;
    }

    Map<List<Integer>, Counter2> table = new HashMap<>();

    void addToTable(List<Integer> input, int expected) {
        if(!table.containsKey(input)) {
            table.put(input, new Counter2());
        }
        table.get(input).add(expected);
    }

    void fit(Sample sample) {
        for(int i=1; i<sample.size(); i++) {
            int start = i - n;
            if(start < 0) {
                start = 0;
            }
            addToTable(sample.input.subList(start, i), sample.expected.get(i));
        }
    }

    int nextToken(List<Integer> context) {
        int start = context.size() - n;
        if(start < 0) {
            start = 0;
        }

        var current = context.subList(start, context.size());
        return table.get(current).draw();

    }
}

