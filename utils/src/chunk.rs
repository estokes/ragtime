use anyhow::Result;
use ragtime::doc::DocStore;

fn main() -> Result<()> {
    const DOC: &str = "/home/eric/Downloads/Fowl Engine.txt";
    let mut store = DocStore::new(128)?;
    for chunk in store.add_document(DOC, Some("A document summary would go here"), 256, 128)? {
        let (id, txt) = chunk?;
        println!("---------------------- CHUNK {id:?} ---------------------\n{txt}");
    }
    Ok(())
}
