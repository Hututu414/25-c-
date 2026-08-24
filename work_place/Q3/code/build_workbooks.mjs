import fs from "node:fs/promises";
import path from "node:path";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [payloadPath, outputRoot, previewRoot] = process.argv.slice(2);
if (!payloadPath || !outputRoot || !previewRoot) {
  throw new Error("usage: node build_workbooks.mjs payload.json output_root preview_root");
}

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
await fs.mkdir(outputRoot, { recursive: true });
await fs.mkdir(previewRoot, { recursive: true });

function columnName(index) {
  let value = index + 1;
  let output = "";
  while (value > 0) {
    value -= 1;
    output = String.fromCharCode(65 + (value % 26)) + output;
    value = Math.floor(value / 26);
  }
  return output;
}

function visualLength(value) {
  return [...String(value ?? "")].reduce(
    (total, character) => total + (/[　-鿿＀-￯]/u.test(character) ? 2 : 1),
    0,
  );
}

function columnWidth(header, rows, index) {
  const sample = rows.slice(0, 250).map((row) => visualLength(row[index]));
  return Math.min(46, Math.max(10, visualLength(header) + 2, ...sample) + 1);
}

function numberFormat(header) {
  if (/^(N|K|Group|组别|人数|孕妇数|.*_N)$/i.test(header)) return "#,##0";
  if (/reliability|proportion|可靠性|比例/i.test(header)) return "0.0000";
  if (/p-value|p_value|Pr\(|risk|风险|cost|error|objective|edf|k-index/i.test(header)) return "0.000000";
  if (/BMI|GA|timing|时点|孕周|delta|shift|cutpoint|Height|AGE/i.test(header)) return "0.0000";
  return null;
}

const qa = [];
for (const specification of payload.workbooks) {
  const workbook = Workbook.create();
  for (const sheetSpec of specification.sheets) {
    const sheet = workbook.worksheets.add(sheetSpec.name.slice(0, 31));
    sheet.showGridLines = false;
    const matrix = [sheetSpec.columns, ...sheetSpec.rows];
    const rowCount = matrix.length;
    const columnCount = sheetSpec.columns.length;
    const fullRange = sheet.getRangeByIndexes(0, 0, rowCount, columnCount);
    fullRange.values = matrix;
    fullRange.format.verticalAlignment = "center";
    const header = sheet.getRangeByIndexes(0, 0, 1, columnCount);
    header.format = {
      fill: "#1F4E78",
      font: { bold: true, color: "#FFFFFF" },
      horizontalAlignment: "center",
      verticalAlignment: "center",
      wrapText: true,
      borders: { preset: "outside", style: "thin", color: "#163A5C" },
    };
    header.format.rowHeight = 28;
    sheet.freezePanes.freezeRows(1);
    for (let column = 0; column < columnCount; column += 1) {
      const range = sheet.getRangeByIndexes(0, column, rowCount, 1);
      range.format.columnWidth = columnWidth(sheetSpec.columns[column], sheetSpec.rows, column);
      const format = numberFormat(sheetSpec.columns[column]);
      if (format && rowCount > 1) {
        sheet.getRangeByIndexes(1, column, rowCount - 1, 1).format.numberFormat = format;
      }
    }
    const inspect = await workbook.inspect({
      kind: "table",
      sheetId: sheet.name,
      range: `A1:${columnName(columnCount - 1)}${Math.min(rowCount, 8)}`,
      include: "values,formulas",
      tableMaxRows: 8,
      tableMaxCols: Math.min(columnCount, 16),
      maxChars: 5000,
    });
    qa.push(inspect.ndjson);
    const preview = await workbook.render({
      sheetName: sheet.name,
      range: `A1:${columnName(columnCount - 1)}${Math.min(rowCount, 25)}`,
      scale: 1,
      format: "png",
    });
    const previewName = `${path.basename(specification.path, ".xlsx")}_${sheet.name}.png`.replaceAll(/[<>:"/\\|?*]/g, "_");
    await fs.writeFile(path.join(previewRoot, previewName), new Uint8Array(await preview.arrayBuffer()));
  }
  const errorScan = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 200 },
    summary: `formula error scan for ${specification.path}`,
  });
  qa.push(errorScan.ndjson);
  const target = path.resolve(outputRoot, specification.path);
  if (!target.startsWith(path.resolve(outputRoot) + path.sep)) {
    throw new Error(`output escaped root: ${target}`);
  }
  await fs.mkdir(path.dirname(target), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(target);
  process.stdout.write(`exported ${specification.path}\n`);
}

await fs.writeFile(path.join(previewRoot, "workbook_inspection.ndjson"), qa.join("\n"), "utf8");
