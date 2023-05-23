<template>
  <div>
    <form @submit.prevent="runScript">
      <label for="question">Question:</label>
      <input id="question" v-model="question" type="text" />
      <button type="submit">Run Script</button>
    </form>
    <div class="result-box" v-if="result">
      <h2>Result:</h2>
      <p>{{ result }}</p>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      result: null,
      question: ''
    };
  },
  methods: {
    async runScript() {
      const response = await axios.post('http://localhost:5001/run_script', {
        question: this.question,
        context: 'your-context-here'
      });
      this.result = response.data;
    },
  },
};
</script>
