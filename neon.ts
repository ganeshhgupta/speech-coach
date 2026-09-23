import { defineConfig } from "@neon/config/v1";

export default defineConfig({
  auth: true,
  buckets: {
    "voice-recordings": { access: "private" },
  },
});
