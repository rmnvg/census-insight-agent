import { describe, expect, it } from "vitest";

import { isNumeric, parseCsv } from "./csv";

describe("parseCsv", () => {
  it("handles quoted fields, escaped quotes, and CRLF", () => {
    expect(parseCsv('Region,"Label, with comma",Value\r\nKarnataka,"say ""hi""",75.36\r\n')).toEqual([
      ["Region", "Label, with comma", "Value"],
      ["Karnataka", 'say "hi"', "75.36"],
    ]);
  });

  it("drops blank trailing lines", () => {
    expect(parseCsv("a,b\n1,2\n\n")).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });

  it("recognises numeric cells", () => {
    expect(["75.36", "6,10,95,297", "-3", "12%"].every(isNumeric)).toBe(true);
    expect(["Karnataka", "2011-12", ""].some(isNumeric)).toBe(false);
  });
});
