import { defineCollection } from "astro:content";
import { docsSchema } from "@astrojs/starlight/schema";
import { glob } from "astro/loaders";

export const collections = {
  docs: defineCollection({
    loader: glob({
      pattern: "**/[^_]*.{md,mdx}",
      base: new URL("../../", import.meta.url),
    }),
    schema: docsSchema(),
  }),
};
