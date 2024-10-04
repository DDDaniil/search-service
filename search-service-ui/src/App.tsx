import React, {useRef, useState} from 'react';
import axios from "axios";
import '@mantine/core/styles.css';
import './App.css';
import {Button, Divider, FileButton, Flex, Group, SimpleGrid, Text, Image} from "@mantine/core";
import {documentTypes} from "./DocumentTypes";


const App = () => {
    const api = "http://127.0.0.1:5000";
    const database = 'http://localhost:8000/';
    const databaseLocal = 'C:\\Users\\danil\\Desktop\\data\\test_data';
    const [predictedCategoryFile, setPredictedCategoryFile] = useState("");
    const [nameCategory, setNameCategory] = useState("");
    const [downloadedFile, setDownloadedFile] = useState<File | null>(null);
    const [files, setFiles] = useState<File[] | null>(null);

    const resetRef = useRef<() => void>(null);

    const clearFile = () => {
        setDownloadedFile(null);
        setPredictedCategoryFile('');
        setNameCategory('');
        setFiles(null);
        resetRef.current?.();
    };

    const handleClickPredictFile = async () => {
        const formData = new FormData();
        if (downloadedFile) {
            formData.append('file', downloadedFile, downloadedFile.name);
        }
        try {
            const response = await axios.post(`${api}/predictFile`, formData, {
               headers: { 'Content-Type': 'multipart/form-data' }
            });
            for (var item of documentTypes) {
                if (item.id === response.data.prediction) {
                    setNameCategory(item.name);
                }
            }
            setPredictedCategoryFile(response.data.prediction);
            await handleGetFiles(`${databaseLocal}/${predictedCategoryFile}`)
        } catch (error) {
            console.error(error);
        }
    };

    const handleGetFiles = async (directory: string) => {
        try {
            const response = await axios.post(`${api}/api/files`,
                {
                    path: directory,
                });
            setFiles(response.data)
            console.log(response.data)
        } catch (error) {
            console.error(error);
        }
    }

    const openNewTab = (category: string, name: string) => {
        const link = document.createElement('a');
        link.href = `${database}${category}/${name}`;
        link.target = '_blank';

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    return (
        <>
            <Image
                radius="md"
                height={150}
                src={'../logo.png'}
                fit={'contain'}
            />
            <h1 style={{textAlign: "center"}}>Сервис поиска тематически схожих документов</h1>
            <div className={'main-body'}>
            <Flex
                mih={50}
                bg="rgba(0, 0, 0, .1)"
                gap="lg"
                justify="center"
                align="center"
                direction="column"
                wrap="wrap"
            >
                <Group>
                    <FileButton onChange={setDownloadedFile} accept="file/pdf">
                        {(props) => <Button {...props}>Загрузка документа</Button>}
                    </FileButton>
                    <Button disabled={!downloadedFile} color="red" onClick={clearFile}>
                        Удалить
                    </Button>
                </Group>
                {downloadedFile && (
                    <Text size="sm" ta="center" mt="sm">
                        Прикрепленный файл: {downloadedFile.name}
                    </Text>
                )}
                <Divider/>
                {downloadedFile ? <Button onClick={handleClickPredictFile}>Сделать предсказание для выбранного документа</Button> : ''}
                {predictedCategoryFile && <p>{"Предугаданная категория математического документа: " + nameCategory + ' (' + predictedCategoryFile + ')' }</p>}
                {predictedCategoryFile && <Button onClick={() => handleGetFiles(`${databaseLocal}/${predictedCategoryFile}`)}>Показать схожие документы</Button>}
                {files && (<SimpleGrid cols={1} spacing="sm" verticalSpacing="sm">
                    {files.map((file) => (
                        <Button
                            color={'green'}
                            key={String(file)}
                            onClick={() => openNewTab(predictedCategoryFile, String(file))}
                        >
                            {String(file)}
                        </Button>
                    ))}
                </SimpleGrid>)}
            </Flex>
        </div>
            <div className={'footer'}>
                <Image
                    radius="md"
                    height={50}
                    src={'../itis.png'}
                    fit={'contain'}
                />
            </div>
        </>
    );
};

export default App;
