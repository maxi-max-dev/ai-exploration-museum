export const chapters = [];
export function chapter(id,title,summary,sections,example,limits,links,questions){chapters.push({id,title,summary,sections,example,limits,experiments:links,questions,prerequisites:id==='C01'?[]:['C'+String(+id.slice(1)-1).padStart(2,'0')]});}
