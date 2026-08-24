import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

if (process.argv.length !== 5) {
  throw new Error("usage: node build_round2_workbooks.mjs payload.json outputs_round2 preview_dir");
}

const [payloadPath, outputRoot, previewRoot] = process.argv.slice(2);
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
await fs.mkdir(outputRoot, { recursive: true });
await fs.mkdir(previewRoot, { recursive: true });

function columnName(index) {
  let result = "";
  for (let value = index + 1; value > 0; value = Math.floor((value - 1) / 26)) {
    result = String.fromCharCode(65 + ((value - 1) % 26)) + result;
  }
  return result;
}

function safeName(value) {
  return value.replace(/[^a-zA-Z0-9_-]+/g, "_");
}

for (const spec of payload.workbooks) {
  const workbook = Workbook.create();
  const previewBook = safeName(path.basename(spec.path, ".xlsx"));
  for (const sheetSpec of spec.sheets) {
    const rows = sheetSpec.rows.length && sheetSpec.rows[0].length
      ? sheetSpec.rows
      : [["status"], ["no records"]];
    const rowCount = rows.length;
    const columnCount = rows[0].length;
    const sheet = workbook.worksheets.add(sheetSpec.name);
    const used = sheet.getRangeByIndexes(0, 0, rowCount, columnCount);
    used.values = rows;
    used.format.font = { name: "Arial", size: 10, color: "#111827" };
    used.format.verticalAlignment = "center";
    const header = sheet.getRangeByIndexes(0, 0, 1, columnCount);
    header.format = {
      fill: "#1F4E78",
      font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
      horizontalAlignment: "center",
      verticalAlignment: "center",
      borders: { preset: "outside", style: "thin", color: "#17365D" },
    };
    header.format.rowHeight = 22;
    sheet.freezePanes.freezeRows(1);
    sheet.showGridLines = false;

    for (let column = 0; column < columnCount; column += 1) {
      const values = rows.slice(0, Math.min(rowCount, 200)).map((row) => String(row[column] ?? ""));
      const width = Math.max(10, Math.min(34, Math.max(...values.map((value) => value.length)) + 2));
      const range = sheet.getRangeByIndexes(0, column, rowCount, 1);
      range.format.columnWidth = width;
      const label = String(rows[0][column]);
      if (/fold|selected_k|history_n|\bN\b|n_obs|n_patients|nfev/i.test(label)) {
        range.format.numberFormat = "0";
      } else if (/pct|percent/i.test(label)) {
        range.format.numberFormat = "0.00";
      } else if (/RMSE|MAE|R2|ICC|prediction|observed|variance|estimate|error|AUC|F1|Precision|Recall|Accuracy|probability|phi|p_value|edf|score|correlation/i.test(label)) {
        range.format.numberFormat = "0.000000";
      }
      if (/warning|error|formula|gam_check|history_row_ids|value/i.test(label)) {
        range.format.wrapText = true;
      }
    }

    const previewRows = Math.min(rowCount, 30);
    const previewRange = `A1:${columnName(columnCount - 1)}${previewRows}`;
    const preview = await workbook.render({
      sheetName: sheetSpec.name,
      range: previewRange,
      scale: 1.1,
      format: "png",
    });
    const previewPath = path.join(previewRoot, `${previewBook}__${safeName(sheetSpec.name)}.png`);
    await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  }

  const firstSheet = spec.sheets[0].name;
  const inspection = await workbook.inspect({
    kind: "table",
    range: `${firstSheet}!A1:L12`,
    include: "values,formulas",
    tableMaxRows: 12,
    tableMaxCols: 12,
    maxChars: 2500,
  });
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: `formula errors in ${spec.path}`,
  });
  console.log(JSON.stringify({ workbook: spec.path, inspection: inspection.ndjson, errors: errors.ndjson }));
  const output = await SpreadsheetFile.exportXlsx(workbook);
  const outputPath = path.join(outputRoot, spec.path);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await output.save(outputPath);
}
