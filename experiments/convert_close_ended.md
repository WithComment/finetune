**Task:** You will be given the content of a CSV file with three columns: `id`, `question` and `answer`. Your task is to process the entire file and append two new columns: `closed_ended_q` and `closed_ended_a`.

**Instructions:**
1.  For each row, you will create a new closed-ended question for the `closed_ended_q` column.
2.  This new question must be formed by rephrasing the original `question` to include the content of the original `answer`.
3.  The correct response to your newly generated question must always be "Yes".
4.  Therefore, the `closed_ended_a` column must **always** be filled with the value "Yes".
5.  Your final output should be the complete CSV data, including the original headers and data, plus the two new columns. Do not provide any other text or explanation.

---

**Example of the Required Transformation:**

**Input CSV Data:**
```csv
id,question,answer
0,What anatomical structure is identified for preservation during dissection?,Lingual nerve.
1,What is being ligated during the surgery?,Blood vessel.
2,How is the patient positioned?,Supine.
```

**Required Output CSV Data:**
```csv
id,question,answer,closed_ended_q,closed_ended_a
0,What anatomical structure is identified for preservation during dissection?,Lingual nerve.,Is the lingual nerve the anatomical structure identified for preservation during dissection?,Yes
1,What is being ligated during the surgery?,Blood vessel.,Is a blood vessel being ligated during the surgery?,Yes
2,How is the patient positioned?,Supine.,Is the patient positioned supine?,Yes
```

---

**Now, apply this transformation to the provided CSV data. Your response should be only the updated CSV content.**
